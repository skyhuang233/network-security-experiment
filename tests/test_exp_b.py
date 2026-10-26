"""Tests for experiment B: capacity calibration driver (§8.1, §9)."""

import pytest

from network_security_experiment.experiments.exp_b import (
    achievable_capacity,
    calibrate,
)


def _quick(points):
    return [p for p in points if p.offered_bytes <= 64]


def test_calibration_monotone_for_m1(tmp_path):
    points = calibrate(
        methods=["M1"],
        payload_gradient=[8, 32, 64],
        max_rounds=10,
        workspace=tmp_path,
    )
    assert len(points) == 3
    assert all(p.delivered for p in points)
    assert all(p.ber == 0.0 for p in points)
    # one-round visibility: delivery latency in rounds is exactly 1 for M1
    assert all(p.latency_rounds == 1 for p in points)
    # more bytes -> more bits per round
    by_size = {p.offered_bytes: p.bits_per_round for p in points}
    assert by_size[64] > by_size[8]


def test_calibration_m2_same_shape(tmp_path):
    points = calibrate(
        methods=["M2"],
        payload_gradient=[16, 64],
        max_rounds=10,
        workspace=tmp_path,
    )
    assert all(p.delivered for p in points)


def test_calibration_m3_k_sweep_feasibility(tmp_path):
    """K=16 is structurally infeasible (reserved header keys); larger K
    deliver within a small round budget; smaller K needs more rounds."""
    points = calibrate(
        methods=["M3"],
        payload_gradient=[16],
        max_rounds=60,
        workspace=tmp_path,
    )
    by_k = {p.key_set_size: p for p in points}
    assert "infeasible" in by_k[16].note
    assert by_k[16].delivered is False
    assert by_k[128].delivered is True
    assert by_k[64].delivered is True
    # structural property: larger K delivers in fewer rounds
    assert by_k[128].latency_rounds <= by_k[64].latency_rounds
    assert by_k[64].latency_rounds >= 2  # multi-segment stream takes rounds


def test_achievable_capacity_picks_max_size():
    points = [
        p for p in calibrate(methods=["M1"], payload_gradient=[8, 64, 512], max_rounds=10)
    ]
    cap = achievable_capacity(points)
    assert cap["M1"]["max_offered_bytes"] == 512
    assert cap["M1"]["ber"] == 0.0
    # repeatable rate: 3 episodes x (512*8) bits over 3 x 2 occupied rounds
    assert cap["M1"]["bits_per_round"] == 512 * 8 / 2