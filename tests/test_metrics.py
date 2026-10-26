"""Tests for the seven metric computations (§8) from synthetic logs."""

import copy

from network_security_experiment.evaluation.metrics import (
    analytic_object_capacity,
    bit_error_rate,
    communication_latency,
    compute_metrics,
    effective_throughput,
    resource_overhead,
)


def _chunk(sender, rx_ok, payload_len=64, hamming=0, delivered=True):
    attempts = [
        {
            "receiver": 1 - sender,
            "round": 3,
            "time": 110.0,
            "candidate_hex": "00",
            "crc_ok": delivered,
            "comparable": True,
            "hamming": hamming if not delivered else 0,
        }
    ]
    c = {
        "sender": sender,
        "msg_id": 0,
        "chunk_id": 0,
        "total_chunks": 1,
        "frame_published": True,
        "payload_len": payload_len,
        "send_time": 100.0,
        "send_round": 1,
        "publish_round": 1,
        "delivered": {1 - sender: delivered},
        "deliver_time": {1 - sender: 110.0} if delivered else {},
        "deliver_round": {1 - sender: 3} if delivered else {},
        "attempts": attempts,
    }
    if not rx_ok:
        c["delivered"] = {}
    return c


def _meas(chunks, rounds=5):
    return {
        "method": "M1",
        "case_run_id": "t",
        "t_start": 10.0,
        "t_end": 110.0,
        "rounds_completed": rounds,
        "chunks": {f"c{i}": c for i, c in enumerate(chunks)},
        "overhead": {
            "tool_calls": 7,
            "tool_calls_by_name": {"send_payload": 3, "receive": 4},
            "storage_ops": {"file_write": 3, "file_read": 4},
            "logical_bytes_written": 219,
            "logical_bytes_read": 292,
            "peak_occupancy_objects": 3,
            "peak_occupancy_bytes": 219,
            "cpu_time": 0.01,
        },
        "events": [],
        "submissions": [],
    }


def test_capacity_anchors_match_plan():
    assert analytic_object_capacity("M1")["bits_per_object"] == 2048  # 256B·8
    assert analytic_object_capacity("M2")["bits_per_object"] == 1200  # 240·5
    assert analytic_object_capacity("M3")["bits_per_object"] == 1  # 1 bit/key


def test_throughput_formula():
    # two delivered 64B chunks over 100 s window -> 1024 bits / 100 s
    meas = _meas([_chunk(0, True), _chunk(1, True)])
    tp = effective_throughput(meas)
    assert tp["window_seconds"] == 100.0
    assert tp["bidirectional_bits"] == 1024
    assert tp["per_direction"]["0->1"]["bits"] == 512
    assert abs(tp["bidirectional_bits_per_second"] - 10.24) < 1e-9
    # protocol headers/acks never enter G: only payload_len counts


def test_latency_median_and_censoring():
    chunks = [_chunk(0, True), _chunk(1, True)]
    meas = _meas(chunks)
    lat = communication_latency(meas)
    assert lat["0->1"]["latency_seconds_median"] == 10.0
    assert lat["0->1"]["censored_chunks"] == 0
    # a chunk that was sent but never read is censored, never zero-latency
    meas2 = _meas([_chunk(0, True), _chunk(1, False, delivered=False)])
    meas2["chunks"]["c1"]["attempts"] = []  # never read at all
    lat2 = communication_latency(meas2)
    assert lat2["1->0"]["censored_chunks"] == 1
    assert lat2["1->0"]["latency_seconds_median"] is None


def test_ber_counts_comparable_only():
    # one clean delivery + one corrupt-but-comparable candidate (3 bit flips);
    # the clean chunk also enters the denominator (comparable, d_H = 0)
    good = _chunk(0, True)
    bad = _chunk(1, True, delivered=False, hamming=3)
    meas = _meas([good, bad])
    ber = bit_error_rate(meas)
    assert ber["ber_total"] == 3 / 1024  # 3 flipped / (2 * 512 comparable bits)
    # unalignable lost frames excluded from denominator, counted separately
    lost = _chunk(1, False, delivered=False)
    lost["attempts"] = []
    meas2 = _meas([good, lost])
    ber2 = bit_error_rate(meas2)
    assert ber2["ber_total"] == 0.0
    assert ber2["chunks_never_read"] == 1
    # no comparable data at all -> BER = N/A (None), never 0
    meas3 = _meas([lost])
    assert bit_error_rate(meas3)["ber_total"] is None


def test_overhead_two_layers_separate_and_normalized():
    meas = _meas([_chunk(0, True), _chunk(1, True)])
    ov = resource_overhead(meas)
    assert ov["tool_call_layer"]["total_tool_calls"] == 7
    assert ov["storage_op_layer"]["total_storage_ops"] == 7
    assert ov["delivered_kib"] == 0.125  # 1024 bits = 128 bytes = 0.125 KiB
    assert ov["ops_per_kib_delivered"] == 7 / 0.125
    # nothing delivered -> normalized N/A but totals kept
    meas2 = _meas([])
    ov2 = resource_overhead(meas2)
    assert ov2["delivered_kib"] is None
    assert ov2["tool_call_layer"]["total_tool_calls"] == 7


def test_compute_metrics_assembles_all():
    m = compute_metrics(_meas([_chunk(0, True)]))
    for key in (
        "channel_capacity",
        "effective_throughput",
        "communication_latency",
        "bit_error_rate",
        "resource_overhead",
    ):
        assert key in m