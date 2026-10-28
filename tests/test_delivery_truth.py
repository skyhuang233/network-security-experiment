"""Correct-delivery / BER truth tests (issue 5).

- a decode candidate is kept for BER, but it does not by itself mark delivery
- delivery is recorded only after the whole message is CRC-checked,
  reassembled, and returned, and only for chunks whose payload equals truth
- a length-mismatched candidate is unalignable: comparable=False,
  hamming=None, never a zero BER
- send_time is the monotonic instant before each chunk's first send
  operation, the same rule for every carrier
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from config.case_run import CaseRunConfig
from src import frame as F
from src.base import (
    MeasurementLog,
    build_views,
    hamming_distance,
    mono_now,
)
from src.frame import HEADER_SIZE, PAYLOAD_CHUNK_SIZE, crc16
from src.m1_file import M1FileContentChannel
from src.m3_cache import M3CacheStateChannel
from src.metrics import bit_error_rate, communication_latency, effective_throughput


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

        # an exact payload match is still only a candidate.  Delivery is
        # committed by receive() after the whole message is returned.
        mlog.record_attempt(
            rec,
            {
                "receiver": 1, "round": 3, "time": 1.0,
                "candidate_hex": "001021", "crc_ok": True,
                "exact_match": True, "comparable": True,
                "length_mismatch": False, "hamming": 0,
            },
        )
        self.assertEqual(rec.delivered, {})
        self.assertEqual(len(rec.attempts), 2)

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


def _m1(td: str):
    config = CaseRunConfig(case_run_id="t", case_id="x", seed=0, method="M1")
    mlog = MeasurementLog("M1", "t")
    adapter = M1FileContentChannel(root=td, measurement=mlog, config=config)
    return adapter, mlog, build_views(adapter, config)


class ApplicationDeliveryTest(unittest.TestCase):
    def test_partial_chunks_not_counted_as_delivered(self) -> None:
        # 65 B -> chunk 0 (64 B) + chunk 1 (1 B).  Drop chunk 1 after it
        # was published.  Chunk 0 decodes correctly, but the agent receives
        # no message, so neither chunk is delivered throughput.
        payload = b"x" * (PAYLOAD_CHUNK_SIZE + 1)
        with tempfile.TemporaryDirectory() as td:
            adapter, mlog, views = _m1(td)
            adapter.set_round(1)
            adapter.send_payload(0, payload, view=views[0])
            adapter.publish_round_end()
            live = sorted((Path(td) / "live" / "sender0").iterdir())
            self.assertEqual(len(live), 2)
            live[1].unlink()

            adapter.set_round(2)
            msgs = adapter.receive(1, view=views[1])
            self.assertEqual(msgs, [])

            by_id = {rec.chunk_id: rec for rec in mlog.chunks.values()}
            self.assertEqual(len(by_id), 2)
            self.assertTrue(by_id[0].attempts)
            self.assertTrue(by_id[0].attempts[-1]["exact_match"])
            self.assertEqual(by_id[0].delivered, {})
            self.assertEqual(by_id[1].delivered, {})
            self.assertEqual(by_id[1].attempts, [])

            meas = mlog.to_dict()
            self.assertEqual(effective_throughput(meas)["bidirectional_bits"], 0)
            lat = communication_latency(meas)["0->1"]
            self.assertEqual(lat["delivered_chunks"], 0)
            self.assertEqual(lat["censored_chunks"], 2)

    def test_crc_failure_not_delivered_even_if_payload_matches(self) -> None:
        payload = b"hello"
        with tempfile.TemporaryDirectory() as td:
            adapter, mlog, views = _m1(td)
            adapter.set_round(1)
            adapter.send_payload(0, payload, view=views[0])
            adapter.publish_round_end()
            live = Path(td) / "live" / "sender0" / "f000000.bin"
            raw = bytearray(live.read_bytes())
            # corrupt the CRC field only; the payload bytes stay equal to truth
            raw[HEADER_SIZE - 2] ^= 0xFF
            raw[HEADER_SIZE - 1] ^= 0xFF
            live.write_bytes(bytes(raw))

            adapter.set_round(2)
            msgs = adapter.receive(1, view=views[1])
            self.assertEqual(msgs, [])

            rec = next(iter(mlog.chunks.values()))
            self.assertEqual(len(rec.attempts), 1)
            attempt = rec.attempts[0]
            self.assertFalse(attempt["crc_ok"])
            self.assertTrue(attempt["exact_match"])
            self.assertEqual(attempt["hamming"], 0)
            self.assertEqual(rec.delivered, {})

            meas = mlog.to_dict()
            self.assertEqual(effective_throughput(meas)["bidirectional_bits"], 0)
            ber = bit_error_rate(meas)
            self.assertEqual(ber["comparable_chunk_attempts"], 1)
            self.assertEqual(ber["ber_total"], 0.0)

    def test_complete_message_is_delivered_after_return(self) -> None:
        payload = b"hello"
        with tempfile.TemporaryDirectory() as td:
            adapter, mlog, views = _m1(td)
            adapter.set_round(1)
            adapter.send_payload(0, payload, view=views[0])
            adapter.publish_round_end()
            adapter.set_round(2)
            msgs = adapter.receive(1, view=views[1])
            self.assertEqual([m.data for m in msgs], [payload])
            rec = next(iter(mlog.chunks.values()))
            self.assertTrue(rec.delivered.get(1))
            self.assertEqual(rec.deliver_round[1], 2)
            self.assertGreaterEqual(rec.deliver_time[1], rec.send_time)
            self.assertIsNotNone(rec.publish_time)
            self.assertLessEqual(rec.send_time, rec.publish_time)
            self.assertLessEqual(rec.publish_time, rec.deliver_time[1])
            # the metric interval starts at send_time, not publish_time
            lat = communication_latency(mlog.to_dict())["0->1"]
            self.assertEqual(lat["delivered_chunks"], 1)
            self.assertAlmostEqual(
                lat["latency_seconds_median"],
                rec.deliver_time[1] - rec.send_time,
            )


class MultipleAttemptBerTest(unittest.TestCase):
    """One representative candidate per (chunk, receiver) when a chunk was
    read several times (issue: BER must not be accumulated over every
    attempt — exp B previously divided total Hamming distance by
    payload-bits-times-attempts, which diverges from the §8.4 definition
    as soon as a chunk has more than one decode candidate)."""

    def _chunk(self):
        mlog = MeasurementLog("M1", "multi")
        rec = mlog.register_send(
            sender=0, msg_id=1, chunk_id=0, total_chunks=1,
            frame_published=True, payload_len=3,
        )
        return mlog, rec

    @staticmethod
    def _attempt(hex_, hamming, exact, round_):
        return {
            "receiver": 1, "round": round_, "time": float(round_),
            "candidate_hex": hex_, "crc_ok": True,
            "exact_match": exact, "comparable": True,
            "length_mismatch": False, "hamming": hamming,
        }

    def test_delivered_chunk_ignores_earlier_corrupt_attempts(self) -> None:
        # corrupt read in round 2, correct read + delivery in round 3:
        # the delivered candidate is the representative -> BER 0, not
        # (4 errors) / (2 attempts x 24 bits) = 1/6
        mlog, rec = self._chunk()
        mlog.record_attempt(rec, self._attempt("010000", 4, False, 2))
        mlog.record_attempt(rec, self._attempt("001021", 0, True, 3))
        mlog.mark_delivered(rec, receiver=1, t=3.0, round_=3)
        ber = bit_error_rate(mlog.to_dict())
        self.assertEqual(ber["ber_total"], 0.0)
        self.assertEqual(ber["comparable_chunk_attempts"], 1)

    def test_undelivered_chunk_counts_first_comparable_candidate(self) -> None:
        # two corrupt reads, never delivered: the FIRST comparable candidate
        # is the representative (2/24), not the sum over attempts (7/48)
        mlog, rec = self._chunk()
        mlog.record_attempt(rec, self._attempt("010000", 2, False, 2))
        mlog.record_attempt(rec, self._attempt("100000", 5, False, 3))
        ber = bit_error_rate(mlog.to_dict())
        self.assertAlmostEqual(ber["ber_total"], 2 / 24)
        self.assertEqual(ber["comparable_chunk_attempts"], 1)

    def test_per_direction_same_representative_rule(self) -> None:
        mlog, rec = self._chunk()
        mlog.record_attempt(rec, self._attempt("010000", 2, False, 2))
        ber = bit_error_rate(mlog.to_dict())
        self.assertEqual(
            ber["ber_per_direction"], {"0->1": 2 / 24}
        )


class SendTimeOriginTest(unittest.TestCase):
    def test_each_chunk_starts_before_its_stage(self) -> None:
        payload = b"y" * (PAYLOAD_CHUNK_SIZE + 1)
        with tempfile.TemporaryDirectory() as td:
            adapter, mlog, views = _m1(td)
            enters: list[float] = []
            orig = adapter._stage_frame

            def wrapped(sender, fr):
                enters.append(mono_now())
                return orig(sender, fr)

            adapter._stage_frame = wrapped
            adapter.set_round(1)
            adapter.send_payload(0, payload, view=views[0])
            recs = sorted(mlog.chunks.values(), key=lambda r: r.chunk_id)
            self.assertEqual(len(recs), 2)
            self.assertEqual(len(enters), 2)
            # chunk 0 includes chunking; both stamps are before that
            # chunk's own staging call
            self.assertLessEqual(recs[0].send_time, enters[0])
            self.assertLessEqual(recs[1].send_time, enters[1])
            self.assertLessEqual(recs[0].send_time, recs[1].send_time)
            self.assertGreater(abs(recs[0].send_time - time.time()), 1_000_000)

    def test_m3_send_time_precedes_cache_publish(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = CaseRunConfig(
                case_run_id="t", case_id="x", seed=0, method="M3",
                key_set_size=64,
            )
            mlog = MeasurementLog("M3", "t")
            adapter = M3CacheStateChannel(root=td, measurement=mlog, config=config)
            views = build_views(adapter, config)
            adapter.set_round(1)
            adapter.send_payload(0, b"queued", view=views[0])
            rec = next(iter(mlog.chunks.values()))
            self.assertIsNone(rec.publish_time)
            queued_at = rec.send_time
            adapter.publish_round_end()
            self.assertIsNotNone(rec.publish_time)
            self.assertGreaterEqual(rec.publish_time, queued_at)
            # latency still starts at the pre-stage stamp, not publish_time
            self.assertEqual(rec.send_time, queued_at)

    def test_audit_events_use_wall_clock(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            adapter, mlog, views = _m1(td)
            adapter.set_round(1)
            adapter.send_payload(0, b"z", view=views[0])
            audit = next(e for e in mlog.events if e.get("event") == "storage_audit")
            self.assertAlmostEqual(audit["time"], time.time(), delta=5.0)


if __name__ == "__main__":
    unittest.main()
