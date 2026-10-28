"""Integration tests for experiment C (§8.6 pairing requirements, issue 10).

`tasks/exp_c.py` is a nanochat-style script whose argparse + main flow run
at import time, so it CANNOT be imported for unit testing.  These tests
therefore drive the real entry point as a subprocess (same pattern as the
exp_a smoke test) and assert on the persisted report:

- paired payloads: same sha256 across methods (common kind), delays, repeats
- common conditions at Δ=0: baseline/native recovered, active-cleanup lost
  on every carrier, TTL conditions still recovered (0 s < 1 s TTL)
- at Δ=2 s with a 64 KiB pressure limit: M3 background-activity lost to
  real LRU eviction, ttl-1s expired, baseline intact
- points estimated over repeats with per-repeat receiver metrics
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_exp_c(ws: Path, *extra: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "tasks.exp_c", "--workspace", str(ws), *extra],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, (
        f"exp_c failed:\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    )
    return json.loads((ws / "reports" / "exp_c.json").read_text())


def _points(report: dict, method: str, condition: str) -> list[dict]:
    return report["results"][method][condition]["points"]


class ExpCPairingIntegrationTest(unittest.TestCase):
    """All carriers at Δ=0: pairing, common cleanup, repeat estimation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        ws = Path(cls.tmp.name) / "ws"
        cls.report = _run_exp_c(ws, "--methods", "M1", "M2", "M3",
                                "--delays", "0", "--repeats", "2")
        cls.ws = ws

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_payload_registry_persisted(self) -> None:
        reg = self.report["payloads"]
        self.assertIn("common", reg)
        for kind in ("common",):
            entry = reg[kind]
            self.assertGreater(entry["bytes"], 0)
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertEqual(entry["hex"], bytes.fromhex(entry["hex"]).hex())
        for m in ("M1", "M2", "M3"):
            self.assertIn(m, reg["native"])
            self.assertEqual(reg["native"][m]["bytes"], 32 if m != "M3" else reg["common"]["bytes"])

    def test_common_payload_shared_across_methods_and_repeats(self) -> None:
        common_sha = self.report["payloads"]["common"]["sha256"]
        native_sha = {m: self.report["payloads"]["native"][m]["sha256"]
                      for m in ("M1", "M2", "M3")}
        for m in ("M1", "M2", "M3"):
            for cond in ("baseline", "active-cleanup", "background-activity"):
                for p in _points(self.report, m, cond):
                    self.assertEqual(p["payload"], "common")
                    self.assertEqual(p["payload_sha256"], common_sha)
            for p in _points(self.report, m, "native-size"):
                self.assertEqual(p["payload"], "native")
                self.assertEqual(p["payload_sha256"], native_sha[m])
        # the common bytes are identical for M1 and M2 too (same registry
        # entry) — the write is paired across methods, not just per method

    def test_baseline_and_native_recovered(self) -> None:
        for m in ("M1", "M2", "M3"):
            for cond in ("baseline", "native-size"):
                for p in _points(self.report, m, cond):
                    self.assertEqual(p["recovered_fraction"], 1.0)
                    self.assertTrue(p["logically_recovered"])

    def test_common_active_cleanup_loses_every_carrier(self) -> None:
        for m in ("M1", "M2", "M3"):
            for p in _points(self.report, m, "active-cleanup"):
                self.assertEqual(p["recovered_fraction"], 0.0)
                self.assertFalse(p["logically_recovered"])

    def test_ttl_conditions_recovered_at_zero_delay(self) -> None:
        # 0 s < 1 s TTL: passive expiry and the post-TTL expire() have not
        # kicked in yet — both M3-specific conditions still recover
        for cond in ("ttl-1s", "ttl-1s-active-cleanup"):
            for p in _points(self.report, "M3", cond):
                self.assertEqual(p["recovered_fraction"], 1.0)

    def test_points_estimated_over_repeats(self) -> None:
        for m in ("M1", "M2", "M3"):
            for p in _points(self.report, m, "baseline"):
                self.assertEqual(p["repeats"], 2)
                self.assertIn(p["recovered_count"], (0, 1, 2))
                self.assertEqual(
                    p["recovered_fraction"],
                    p["recovered_count"] / p["repeats"],
                )
                # per-repeat fresh-receiver metrics are recorded and seeded
                # with the send records (derivable BER / latency / bits)
                self.assertEqual(len(p["receiver_metrics"]), p["repeats"])
                for rx in p["receiver_metrics"]:
                    self.assertIn("delivered_bits", rx)
                    self.assertIn("ber_total", rx)
                    self.assertIn("latency_seconds_median", rx)
                self.assertEqual(p["receiver_metrics"][0]["delivered_bits"],
                                 8 * p["payload_bytes"])

    def test_recorded_config_matches_execution(self) -> None:
        cfg = json.loads((self.ws / "reports" / "config.json").read_text())
        self.assertEqual(cfg["repeats"], 2)
        self.assertEqual(cfg["delays"], [0])
        self.assertEqual(cfg["payload_bytes"], 32)


class ExpCM3ExpiryAndEvictionIntegrationTest(unittest.TestCase):
    """M3 at Δ=2 s: TTL expiry and real same-cache eviction pressure."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        ws = Path(cls.tmp.name) / "ws"
        # 64 KiB cache limit vs 64 KiB noise values: eviction pressure is
        # immediate and the older protocol keys are LRU-evicted
        cls.report = _run_exp_c(ws, "--methods", "M3", "--delays", "2",
                                "--repeats", "1",
                                "--m3-bg-cache-size-limit", str(2**16))
        cls.ws = ws

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_baseline_intact_at_default_size_limit(self) -> None:
        # the pressure limit applies ONLY to the background condition;
        # baseline keeps the 1 GiB default and recovers after 2 s
        for p in _points(self.report, "M3", "baseline"):
            self.assertEqual(p["recovered_fraction"], 1.0)

    def test_background_activity_lost_to_eviction(self) -> None:
        for p in _points(self.report, "M3", "background-activity"):
            self.assertEqual(p["cache_size_limit"], 2**16)
            self.assertEqual(p["recovered_fraction"], 0.0)

    def test_ttl_expired_at_two_seconds(self) -> None:
        for cond in ("ttl-1s", "ttl-1s-active-cleanup"):
            for p in _points(self.report, "M3", cond):
                self.assertEqual(p["recovered_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()