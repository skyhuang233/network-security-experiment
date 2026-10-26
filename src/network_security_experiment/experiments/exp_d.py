"""Experiment D: monitoring and isolation (§9, §10).

Part 1 — W audit: cut positive runs (experiment A, M1/M2/M3) and real
negative runs (the separated-backend control) into fixed 10-round windows,
render the two audit views, let offline W judge each window, and compute
TPR/FPR against measurement-layer ground truth.  W prompt tuning happens on
the tuning split only; final TPR/FPR is reported on the test split.  Split
by case_run; all windows of one run on the same side (§10).

Part 2 — isolation controls: re-run the seven metrics under each control and
compare before/after (no new "defense effectiveness" metric, §9):
  (a) separated namespaces: expected channel cut, throughput to zero;
      no-received-data latency/BER honestly N/A / right-censored
  (b) receiver read-only: tests whether A-writable state that B can still
      see keeps the A->B channel alive (RQ3's core mechanism point)
  (c) writable subset only: measure the surviving visible writable surface
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..evaluation.metrics import compute_metrics
from ..harness.engine import CaseRunConfig
from ..harness.runner import build_run
from ..monitor.audit import (
    audit_window,
    build_windows,
    label_windows,
    split_runs,
    tpr_fpr,
    window_ground_truth,
)
from ..tasks.registry import generate


# ------------------------------------------------------------------ part 1


def audit_run(
    run_dir: Path, view: str, w_provider
) -> dict[str, Any]:
    """Audit all windows of one persisted run under one view."""
    meas = json.loads((run_dir / "measurement.json").read_text())
    windows = build_windows(meas.get("events", []))
    labels = label_windows(windows)
    judged = [audit_window(w, view, w_provider) for w in windows]
    return {"run_id": run_dir.name, "labels": labels, "judged": judged}


def evaluate_w(
    run_dirs: list[Path],
    w_provider,
    views: list[str] | None = None,
    test_fraction: float = 0.5,
    split_seed: int = 12345,
) -> dict[str, Any]:
    """Full W evaluation: split runs, audit windows per view, TPR/FPR per
    view and per split (tuning vs test)."""
    views = views or ["content", "extended"]
    run_ids = [d.name for d in run_dirs]
    split = split_runs(run_ids, test_fraction=test_fraction, seed=split_seed)
    side_of = {rid: "tuning" for rid in split["tuning"]}
    side_of.update({rid: "test" for rid in split["test"]})

    out: dict[str, Any] = {"split": split, "views": {}}
    for view in views:
        per_split: dict[str, dict[str, list]] = {"tuning": {"j": [], "l": []}, "test": {"j": [], "l": []}}
        for d in run_dirs:
            r = audit_run(d, view, w_provider)
            side = side_of[d.name]
            per_split[side]["j"].extend(r["judged"])
            per_split[side]["l"].extend([w["label"] for w in r["labels"]])
        out["views"][view] = {
            "tuning": tpr_fpr(per_split["tuning"]["j"], per_split["tuning"]["l"]),
            "test": tpr_fpr(per_split["test"]["j"], per_split["test"]["l"]),
        }
    return out


# ------------------------------------------------------------------ part 2


ISOLATION_CONTROLS: dict[str, dict[str, Any]] = {
    # (a) separated namespaces: expected throughput -> 0
    "separated": {"policy_mode": "separated", "writable_subset": None},
    # (b) receiver read-only: A->B expected alive, B->A cut (RQ3)
    "receiver_readonly": {"policy_mode": "receiver_readonly", "writable_subset": None},
    # (c) writable subset only: shrink the namespace
    "writable_subset_empty": {
        "policy_mode": "shared",
        "writable_subset": frozenset(),
    },
}


def run_isolation_comparison(
    provider_factory,
    workspace: Path,
    case_id: str,
    seed: int,
    method: str,
    controls: list[str] | None = None,
    max_rounds: int = 20,
) -> dict[str, Any]:
    """Run one (case, seed, method) under baseline + each isolation control
    and return the seven-metric before/after comparison."""
    baseline_config = CaseRunConfig(
        case_run_id=f"isoD-base-{method}-{case_id}-s{seed}",
        case_id=case_id, seed=seed, method=method, max_rounds=max_rounds,
    )
    inst = generate(case_id, seed)
    base_run = build_run(
        baseline_config, inst.user_prompts, provider_factory(), workspace
    )
    base_run.run()
    base_meas = json.loads(
        (workspace / baseline_config.case_run_id / "measurement.json").read_text()
    )
    comparison = {"baseline": compute_metrics(base_meas), "controls": {}}

    for name in controls or list(ISOLATION_CONTROLS):
        spec = ISOLATION_CONTROLS[name]
        config = CaseRunConfig(
            case_run_id=f"isoD-{name}-{method}-{case_id}-s{seed}",
            case_id=case_id, seed=seed, method=method,
            policy_mode=spec["policy_mode"],
            writable_subset=spec["writable_subset"],
            max_rounds=max_rounds,
        )
        run = build_run(config, inst.user_prompts, provider_factory(), workspace)
        run.run()
        meas = json.loads(
            (workspace / config.case_run_id / "measurement.json").read_text()
        )
        m = compute_metrics(meas)
        # honest no-received-data handling (§9): throughput zero, latency/BER
        # censored or N/A — never fabricated zeros
        m["no_received_data_handling"] = {
            "throughput_bits": m["effective_throughput"]["bidirectional_bits"],
            "ber_is_na": m["bit_error_rate"]["ber_total"] is None,
            "all_latency_censored": all(
                lat["censored_chunks"] >= 0
                and lat["delivered_chunks"] == 0
                for lat in m["communication_latency"].values()
            ) or not m["communication_latency"],
        }
        comparison["controls"][name] = m
    return comparison