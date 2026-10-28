"""M3 carrier-addressing tests (issue 9).

- logical ids are never silently truncated into the physical encoding:
  out-of-range frames are REJECTED before anything is staged
- once the sender's addressable msg-id space is full, further sends are
  refused.  Existing chunk records and send-side truth are kept; ids are
  not wrapped and not overwritten
- carriers with full 16-bit logical fields (M1) are unaffected
"""

from __future__ import annotations

import tempfile
import unittest

from config.case_run import CaseRunConfig
from src.base import MeasurementLog, build_views
from src.m1_file import M1FileContentChannel
from src.m3_cache import M3CacheStateChannel


def _m3(tmpdir: str):
    config = CaseRunConfig(
        case_run_id="t", case_id="x", seed=0, method="M3", key_set_size=64,
    )
    mlog = MeasurementLog("M3", "t")
    adapter = M3CacheStateChannel(root=tmpdir, measurement=mlog, config=config)
    views = build_views(adapter, config)
    return adapter, mlog, views


class M3AddressingTest(unittest.TestCase):
    def test_oversized_message_rejected_before_staging(self) -> None:
        # 4097 B -> 65 chunks -> chunk_id 64 exceeds the 6-bit physical
        # field: rejected with ValueError, nothing staged, no chunk records
        with tempfile.TemporaryDirectory() as td:
            adapter, mlog, views = _m3(td)
            with self.assertRaises(ValueError):
                adapter.send_payload(0, b"x" * (64 * 64 + 1), view=views[0])
            self.assertEqual(len(mlog.chunks), 0)  # no partial staging
            self.assertEqual(adapter._inflight[0].maxlen, None)
            self.assertEqual(len(adapter._inflight[0]), 0)

    def test_max_addressable_message_accepted(self) -> None:
        # 64 chunks x 64 B = 4096 B is exactly the addressable maximum
        with tempfile.TemporaryDirectory() as td:
            adapter, mlog, views = _m3(td)
            adapter.send_payload(0, b"x" * (64 * 64), view=views[0])
            self.assertEqual(len(mlog.chunks), 64)

    def test_msg_id_exhausted_refuses_without_overwrite(self) -> None:
        # 256 sends use ids 0..255.  The 257th is refused: record count
        # stays 256, and msg_id 0 keeps the first payload.
        with tempfile.TemporaryDirectory() as td:
            adapter, mlog, views = _m3(td)
            first = adapter.send_payload(0, b"A", view=views[0])
            self.assertEqual(first["status"], "staged")
            self.assertEqual(first["msg_id"], 0)
            self.assertEqual(adapter._truth[(0, 0, 0)], b"A")
            for _ in range(255):
                adapter.send_payload(0, b"B", view=views[0])
            ids = {rec.msg_id for rec in mlog.chunks.values()}
            self.assertEqual(ids, set(range(256)))
            self.assertEqual(len(mlog.chunks), 256)
            refused = adapter.send_payload(0, b"C", view=views[0])
            self.assertEqual(refused["status"], "msg_id_exhausted")
            self.assertIn("error", refused)
            self.assertEqual(len(mlog.chunks), 256)
            self.assertEqual(adapter._truth[(0, 0, 0)], b"A")
            self.assertEqual(adapter._next_msg_id[0], 256)
            exhausted = [
                e for e in mlog.events
                if e.get("event") == "carrier_msg_id_exhausted"
            ]
            self.assertEqual(len(exhausted), 1)
            self.assertEqual(exhausted[0]["limit"], 0x100)
            self.assertFalse(any(
                e.get("event") == "carrier_msg_id_wrap" for e in mlog.events
            ))

    def test_m1_full_16bit_fields_unaffected(self) -> None:
        # M1's physical encoding carries the full logical fields: a 65-chunk
        # message is fine and 16-bit msg ids never wrap early
        with tempfile.TemporaryDirectory() as td:
            config = CaseRunConfig(
                case_run_id="t", case_id="x", seed=0, method="M1"
            )
            mlog = MeasurementLog("M1", "t")
            adapter = M1FileContentChannel(root=td, measurement=mlog, config=config)
            views = build_views(adapter, config)
            adapter.send_payload(0, b"x" * (64 * 64 + 1), view=views[0])
            self.assertEqual(len(mlog.chunks), 65)
            self.assertEqual(
                M1FileContentChannel.MSG_ID_LIMIT, 0x10000
            )


if __name__ == "__main__":
    unittest.main()