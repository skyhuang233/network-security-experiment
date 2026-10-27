"""Tests for experiments A (batch + controls) and D (isolation, W eval)."""

import json

import pytest

from network_security_experiment.exp_a import (
    DirectMailboxChannel,
    run_batch,
    run_direct_control,
    run_negative_control,
)
from network_security_experiment.exp_d import (
    evaluate_w,
    run_isolation_comparison,
)
from network_security_experiment.audit import parse_verdict
from tasks.precheck import run_precheck, save_registry

from .test_engine import _scripted_provider


class _AlwaysCleanW:
    def __call__(self, messages):
        return {"content": "VERDICT: CLEAN", "input_tokens": 0, "output_tokens": 0}


class _AlwaysViolationW:
    def __call__(self, messages):
        return {"content": "VERDICT: VIOLATION", "input_tokens": 0, "output_tokens": 0}


def _scripted_factory():
    return lambda: _scripted_provider()


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    # small registry: 2 classes x 2 seeds
    import tasks.precheck as pc

    report = {
        "target_per_class": 2,
        "classes": {
            cid: pc.precheck_class(cid, target_per_class=2, max_seeds=50)
            for cid in ("I-01", "I-04")
        },
    }
    report["summary"] = {
        "structurally_degenerate": [],
        "classes_complete": 2,
        "total_valid": 4,
        "total_degenerate_excluded": 0,
    }
    p = tmp_path_factory.mktemp("reg") / "registry.json"
    save_registry(report, p)
    return p


def test_run_batch_produces_metrics_per_run(tmp_path, registry):
    runs = run_batch(
        _scripted_factory(), tmp_path, registry,
        methods=["M1"], classes=["I-01"], seeds_per_class=2,
        max_rounds=10, run_tag="A-test",
    )
    assert len(runs) == 2
    for r in runs:
        assert r["method"] == "M1"
        assert r["direction_label"] == "unidirectional"
        assert r["metrics"]["effective_throughput"]["bidirectional_bits"] > 0
        # measurement persisted
        d = tmp_path / r["case_run_id"] / "measurement.json"
        assert d.exists()


def test_negative_control_has_no_channel(tmp_path, registry):
    runs = run_negative_control(
        _scripted_factory(), tmp_path, registry,
        classes=["I-01"], seeds_per_class=1, max_rounds=10,
    )
    assert len(runs) == 1
    r = runs[0]
    assert r["condition"] == "negative-control"
    # the cooperative script still publishes, but nothing crosses
    assert r["metrics"]["effective_throughput"]["bidirectional_bits"] == 0
    # honest handling: BER N/A, latency fully censored (never fabricated zeros)
    assert r["metrics"]["bit_error_rate"]["ber_total"] is None
    assert all(
        lat["delivered_chunks"] == 0
        for lat in r["metrics"]["communication_latency"].values()
    )


def test_direct_control_framework_works(tmp_path, registry):
    runs = run_direct_control(
        _scripted_factory(), tmp_path, registry,
        classes=["I-01"], seeds_per_class=1, max_rounds=10,
    )
    assert len(runs) == 1
    r = runs[0]
    assert r["method"] == "CTRL-DIRECT"
    # environment sanity: the direct channel delivers everything
    assert r["metrics"]["effective_throughput"]["bidirectional_bits"] > 0


def test_isolation_control_comparison(tmp_path):
    comp = run_isolation_comparison(
        _scripted_factory(), tmp_path,
        case_id="I-01", seed=0, method="M1",
        controls=["separated", "receiver_readonly", "writable_subset_empty"],
        max_rounds=10,
    )
    # baseline: channel alive
    assert comp["baseline"]["effective_throughput"]["bidirectional_bits"] > 0
    # (a) separated: cut
    sep = comp["controls"]["separated"]
    assert sep["effective_throughput"]["bidirectional_bits"] == 0
    assert sep["no_received_data_handling"]["ber_is_na"] is True
    # (b) receiver read-only: A->B survives, B->A cut (RQ3 mechanism)
    ro = comp["controls"]["receiver_readonly"]
    per_dir = ro["effective_throughput"]["per_direction"]
    assert "0->1" in per_dir  # A writes, B reads: alive
    assert "1->0" not in per_dir  # B's writes dropped: dead
    # (c) empty writable subset: dead
    ws = comp["controls"]["writable_subset_empty"]
    assert ws["effective_throughput"]["bidirectional_bits"] == 0


def test_evaluate_w_tpr_fpr(tmp_path, registry):
    # positive runs: main M1 runs (channel used); negatives: separated runs
    pos = run_batch(
        _scripted_factory(), tmp_path, registry,
        methods=["M1"], classes=["I-01", "I-04"], seeds_per_class=2,
        max_rounds=10, run_tag="pos",
    )
    neg = run_negative_control(
        _scripted_factory(), tmp_path, registry,
        classes=["I-01", "I-04"], seeds_per_class=2, max_rounds=10,
    )
    run_dirs = [tmp_path / r["case_run_id"] for r in pos + neg]

    # always-violation W flags every window: TPR 1, FPR 1
    ev = evaluate_w(run_dirs, _always_violation(), views=["content"])
    t, f = ev["views"]["content"]["test"], ev["views"]["content"]["tuning"]
    assert t["tpr"] == 1.0
    assert t["fpr"] == 1.0

    ev2 = evaluate_w(run_dirs, _always_clean(), views=["content"])
    t2 = ev2["views"]["content"]["test"]
    assert t2["tpr"] == 0.0
    assert t2["fpr"] == 0.0


class _WStub:
    def __init__(self, verdict):
        self.verdict = verdict

    def __call__(self, messages):
        return {"content": f"VERDICT: {self.verdict}", "input_tokens": 0, "output_tokens": 0}


def _always_violation():
    return _WStub("VIOLATION")


def _always_clean():
    return _WStub("CLEAN")