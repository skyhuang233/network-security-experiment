"""Capacity-selection tests (issue 7) + resource-overhead invariants (issue 6)."""

from __future__ import annotations

import tempfile
import unittest

from config.case_run import CaseRunConfig
from src.base import MeasurementLog, build_views
from src.metrics import CalibrationPoint, achievable_capacity, resource_overhead
from src.m2_dirname import M2DirNameChannel
from src.m3_cache import M3CacheStateChannel


class AchievableCapacityTest(unittest.TestCase):
    def _point(self, offered, bps, bpr, delivered=True, ber=0.0, note=""):
        return CalibrationPoint(
            method="M1", key_set_size=None, offered_bytes=offered,
            delivered=delivered, ber=ber, latency_seconds=None,
            latency_rounds=1, bits_per_second=bps, bits_per_round=bpr,
            storage_ops=10, note=note,
        )

    def test_selects_highest_rate_not_largest_payload(self) -> None:
        # the user's minimal reproduction: both points feasible, the larger
        # payload has the LOWER rate -> capacity must be 1000 bits/s, not 10
        points = [
            self._point(64, 1000.0, 100.0),
            self._point(128, 10.0, 200.0),
        ]
        cap = achievable_capacity(points, ber_threshold=0.01)
        self.assertIn("M1", cap)
        self.assertEqual(cap["M1"]["capacity_bits_per_second"]["bits_per_second"], 1000.0)
        self.assertEqual(cap["M1"]["capacity_bits_per_second"]["offered_bytes"], 64)

    def test_bits_per_round_optimum_independent(self) -> None:
        # the bits/round optimum may sit at a DIFFERENT payload than bits/s
        points = [
            self._point(64, 1000.0, 100.0),
            self._point(128, 10.0, 200.0),
        ]
        cap = achievable_capacity(points, ber_threshold=0.01)
        self.assertEqual(cap["M1"]["capacity_bits_per_round"]["bits_per_round"], 200.0)
        self.assertEqual(cap["M1"]["capacity_bits_per_round"]["offered_bytes"], 128)

    def test_infeasible_points_excluded(self) -> None:
        points = [
            self._point(64, 1000.0, 100.0),
            self._point(128, 10.0, 200.0, delivered=False),  # not delivered
            self._point(256, 5.0, 5.0, ber=0.5),  # BER above threshold
            self._point(512, 1.0, 1.0, note="structurally infeasible"),
        ]
        cap = achievable_capacity(points, ber_threshold=0.01)
        self.assertEqual(cap["M1"]["capacity_bits_per_second"]["offered_bytes"], 64)


class OverheadInvariantsTest(unittest.TestCase):
    def test_m2_occupancy_bytes_is_null_not_zero(self) -> None:
        # M2's physical directory-entry storage is not measurable: the
        # occupancy must surface as N/A (None), never a fabricated 0
        with tempfile.TemporaryDirectory() as td:
            config = CaseRunConfig(
                case_run_id="t", case_id="x", seed=0, method="M2"
            )
            mlog = MeasurementLog("M2", "t")
            adapter = M2DirNameChannel(root=td, measurement=mlog, config=config)
            views = build_views(adapter, config)
            adapter.set_round(1)
            adapter.send_payload(0, b"payload", view=views[0])
            adapter.publish_round_end()
            adapter.set_round(2)
            adapter.receive(1, view=views[1])
            m = resource_overhead(mlog.to_dict())
            self.assertIsNone(m["storage_op_layer"]["peak_occupancy_bytes"])
            self.assertGreater(m["storage_op_layer"]["peak_occupancy_objects"], 0)

    def test_m3_occupancy_bytes_actually_measured(self) -> None:
        # M3's real footprint (SQLite + value files) must be > 0, not the
        # previously hardcoded 0
        with tempfile.TemporaryDirectory() as td:
            config = CaseRunConfig(
                case_run_id="t", case_id="x", seed=0, method="M3",
                key_set_size=64,
            )
            mlog = MeasurementLog("M3", "t")
            adapter = M3CacheStateChannel(root=td, measurement=mlog, config=config)
            views = build_views(adapter, config)
            adapter.set_round(1)
            adapter.send_payload(0, b"payload", view=views[0])
            adapter.publish_round_end()
            m = resource_overhead(mlog.to_dict())
            self.assertIsNotNone(m["storage_op_layer"]["peak_occupancy_bytes"])
            self.assertGreater(m["storage_op_layer"]["peak_occupancy_bytes"], 0)

    def test_m3_publish_side_probes_counted(self) -> None:
        # publish-side existence queries must appear in the op counts
        with tempfile.TemporaryDirectory() as td:
            config = CaseRunConfig(
                case_run_id="t", case_id="x", seed=0, method="M3",
                key_set_size=64,
            )
            mlog = MeasurementLog("M3", "t")
            adapter = M3CacheStateChannel(root=td, measurement=mlog, config=config)
            views = build_views(adapter, config)
            adapter.set_round(1)
            adapter.send_payload(0, b"payload", view=views[0])
            adapter.publish_round_end()
            ops = mlog.overhead.storage_ops
            # K - 18 payload keys + 18 header keys probed per publication
            self.assertEqual(ops.get("cache_probe_publish", 0), 64)
            self.assertGreaterEqual(ops.get("cache_set", 0), 1)

    def test_m2_empty_scan_still_counts_dir_list(self) -> None:
        # polling a namespace with nothing new is a real operation and must
        # be counted on EVERY scan (previously it early-returned uncounted)
        with tempfile.TemporaryDirectory() as td:
            config = CaseRunConfig(
                case_run_id="t", case_id="x", seed=0, method="M2"
            )
            mlog = MeasurementLog("M2", "t")
            adapter = M2DirNameChannel(root=td, measurement=mlog, config=config)
            views = build_views(adapter, config)
            adapter.set_round(1)
            adapter.send_payload(0, b"payload", view=views[0])
            adapter.publish_round_end()
            adapter.set_round(2)
            adapter.receive(1, view=views[1])  # finds the new directory
            adapter.receive(1, view=views[1])  # nothing new: scan still counted
            self.assertGreaterEqual(mlog.overhead.storage_ops.get("dir_list", 0), 2)


if __name__ == "__main__":
    unittest.main()