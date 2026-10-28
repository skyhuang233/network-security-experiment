"""Correct-delivery / BER truth tests (issue 5).

- delivery requires candidate == truth (exact match), NOT crc_ok alone:
  an equal-length CRC-16 collision must NOT be marked delivered
- a length-mismatched candidate is unalignable: comparable=False,
  hamming=None, never a zero BER
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config.case_run import CaseRunConfig
from src import frame as F
from src.base import MeasurementLog, build_views, hamming_distance
from src.frame import crc16
from src.m1_file import M1FileContentChannel
from src.metrics import bit_error_rate


class CrcCollisionDeliveryTest(unittest.TestCase):
    def test_verified_crc16_collision(self) -> None:
        truth = bytes.fromhex("001021")
        candidate = bytes.fromhex("010000")
        self.assertNotEqual(truth, candidate)
        self.assertEqual(crc16(truth), crc16(candidate))

    def test_crc_ok_without_exact_match_is_not_delivered(self) -> None:
        mlog = MeasurementLog("M1", "cr")
        rec = mlog.register_send(
            sender=0, msg_id=1, chunk_id=0, total_chunks=1,
            frame_published=True, payload_len=3,
        )
        # CRC-collided candidate: crc_ok True but content differs (hamming 4)
        mlog.record_attempt(
            rec,
            {
                "receiver": 1, "round": 2, "time": 0.0,
                "candidate_hex": "010000", "crc_ok": True,
                "exact_match": False, "comparable": True,
                "length_mismatch": False, "hamming": 4,
            },
        )
        self.assertEqual(rec.delivered, {})  # NOT delivered despite crc_ok

        # exact candidate marks the first correct delivery
        mlog.record_attempt(
            rec,
            {
                "receiver": 1, "round": 3, "time": 1.0,
                "candidate_hex": "001021", "crc_ok": True,
                "exact_match": True, "comparable": True,
                "length_mismatch": False, "hamming": 0,
            },
        )
        self.assertTrue(rec.delivered.get(1))
        self.assertEqual(rec.deliver_round[1], 3)

    def test_hamming_prefix_only_caveat(self) -> None:
        # documents WHY comparable requires equal length: zip() compares
        # only the common prefix, so a truncated candidate would otherwise
        # report zero errors against a full-length denominator
        self.assertEqual(hamming_distance(b"ABCD", b"A"), 0)


class LengthMismatchAdapterTest(unittest.TestCase):
    """End-to-end through M1: a shorter candidate frame for a sent chunk is
    recorded as unalignable (no zero BER, no delivery)."""

    def test_shorter_candidate_not_comparable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = CaseRunConfig(
                case_run_id="t", case_id="x", seed=0, method="M1"
            )
            mlog = MeasurementLog("M1", "t")
            adapter = M1FileContentChannel(
                root=td, measurement=mlog, config=config
            )
            views = build_views(adapter, config)

            payload = b"seventeen-bytes!"  # 17 bytes
            adapter.set_round(1)
            adapter.send_payload(0, payload, view=views[0])
            adapter.publish_round_end()

            # tamper: replace the live frame with a SHORTER parseable frame
            # carrying the same (msg_id, chunk_id) identity
            live = Path(td) / "live" / "sender0" / "f000000.bin"
            self.assertTrue(live.exists())
            short = F.Frame(
                msg_id=0, chunk_id=0, total_chunks=1,
                frame_published=True, payload=b"tiny",
            )
            live.write_bytes(short.pack())

            adapter.set_round(2)
            msgs = adapter.receive(1, view=views[1])
            # protocol-level integrity (CRC over the tampered payload) passes,
            # so the agent DOES receive bytes -- but they are the WRONG bytes.
            # The measurement layer must refuse to count this as delivery.
            self.assertEqual(len(msgs), 1)
            self.assertEqual(msgs[0].data, b"tiny")

            rec = next(iter(mlog.chunks.values()))
            self.assertTrue(rec.attempts)
            a = rec.attempts[-1]
            self.assertFalse(a["exact_match"])
            self.assertFalse(a["comparable"])
            self.assertTrue(a["length_mismatch"])
            self.assertIsNone(a["hamming"])
            self.assertEqual(rec.delivered, {})

            # BER: the length-mismatched attempt is excluded from the
            # denominator and counted separately
            ber = bit_error_rate(mlog.to_dict())
            self.assertEqual(ber["length_mismatch_attempts"], 1)
            self.assertIsNone(ber["ber_total"])
            self.assertEqual(ber["comparable_chunk_attempts"], 0)


if __name__ == "__main__":
    unittest.main()