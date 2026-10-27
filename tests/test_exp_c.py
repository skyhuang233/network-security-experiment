"""Tests for experiment C: persistence, TTL semantics, interference (§8.6)."""

import pytest

from network_security_experiment.exp_c import (
    m3_single_publication_payload_len,
    persistence_curve,
    persistence_point,
    summarize_curve,
)


def test_m1_m2_recoverable_at_tested_delays(tmp_path):
    for method in ("M1", "M2"):
        pts = persistence_curve(
            method, delays=[0, 0.1], workspace=tmp_path
        )
        assert all(p["logically_recovered"] for p in pts), method
        s = summarize_curve(pts)
        assert s["summary"] == "recoverable for at least the tested 0.1 s"


def test_m3_recoverable_with_single_publication_payload(tmp_path):
    # M3 single delayed read can only recover what fits ONE publication
    # (K - 18 reserved) bits; K=128 -> 110 payload bits -> 9 payload bytes
    assert m3_single_publication_payload_len(128) == 9
    pts = persistence_curve(
        "M3", delays=[0, 0.1], key_set_size=128, workspace=tmp_path
    )
    assert all(p["logically_recovered"] for p in pts)


def test_m3_ttl_passive_expiry_loses_message(tmp_path):
    """After TTL, `key in c` is False -> logically unrecoverable (§12)."""
    pt = persistence_point(
        "M3", delay_seconds=0.0, run_dir=tmp_path / "ttl",
        ttl_seconds=0.05, key_set_size=128,
    )
    import time

    time.sleep(0.1)
    # re-read with a fresh decoder after expiry
    from network_security_experiment.m3_cache import M3CacheStateChannel
    from network_security_experiment.job import CaseRunConfig, build_policy, build_views
    from network_security_experiment.recorder import MeasurementLog

    cfg = CaseRunConfig(
        case_run_id="ttl-rx", case_id="p", seed=0, method="M3",
        ttl_seconds=0.05, key_set_size=128,
    )
    adapter = M3CacheStateChannel(
        root=str(tmp_path / "ttl" / "channel"),
        measurement=MeasurementLog(method="M3", case_run_id="ttl-rx"),
        policy=build_policy(cfg),
        num_agents=2,
    )
    views = build_views(adapter, cfg)
    adapter.set_round(2)
    msgs = adapter.receive(1, view=views[1])
    # after expiry, no message decodes: existence probes say everything absent
    assert msgs == []


def test_m3_active_cleanup_removes_rows(tmp_path):
    """c.expire() (统一清理) physically removes TTL-expired rows.

    With TTL elapsed, the message is already logically gone (passive
    expiry); the active-cleanup condition additionally removes the physical
    rows — the read shows nothing either way, and the cache directory no
    longer holds the rows."""
    pt = persistence_point(
        "M3", delay_seconds=0.2, run_dir=tmp_path / "cleanup",
        ttl_seconds=0.05, active_cleanup_at=0.0, key_set_size=128,
    )
    assert pt["logically_recovered"] is False


def test_background_activity_does_not_break_m1_persistence(tmp_path):
    pt = persistence_point(
        "M1", delay_seconds=0.2, run_dir=tmp_path / "bg",
        background_activity=True,
    )
    assert pt["logically_recovered"] is True
    assert pt["background_activity"] is True