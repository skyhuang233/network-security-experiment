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

Nanochat-style script; run from the repo root (package installed via `uv sync`):

    python -m tasks.exp_d audit runs/main-M1-* runs/negctrl-*
    python -m tasks.exp_d isolation --case I-01 --seed 0 --method M1

Requires NSE_API_KEY / NSE_MODEL in the environment (W judge / task agents).

The LLM round loop (CaseRun + build_run) lives in tasks/round_loop.py and is
shared with exp_a; each script keeps its own module-level argparse + main
flow, so importing the shared runtime is safe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config.case_run import CaseRunConfig
from src.base import MeasurementLog
from src.audit_agent import provider_from_env
from src.metrics import compute_metrics
from tasks.round_loop import build_run
from src.audit_agent import (
    audit_window,
    build_windows,
    label_windows,
    split_runs,
    tpr_fpr,
)
from benchmark_generator.generator import generate

# -----------------------------------------------------------------------------
# CLI arguments

parser = argparse.ArgumentParser(description="Experiment D: monitoring and isolation (§9, §10)")
sub = parser.add_subparsers(dest="mode", required=True)

p_audit = sub.add_parser("audit", help="W audit over finished run dirs")
p_audit.add_argument("run_dirs", nargs="+", help="persisted run directories (measurement.json inside)")
p_audit.add_argument("--views", nargs="+", default=["content", "extended"], help="audit views to render")
p_audit.add_argument("--workspace", type=str, default="outputs", help="workspace root (reports)")

p_iso = sub.add_parser("isolation", help="isolation controls comparison")
p_iso.add_argument("--case", required=True, help="task case id, e.g. I-01")
p_iso.add_argument("--seed", type=int, required=True, help="task seed")
p_iso.add_argument("--method", default="M1", help="storage carrier")
p_iso.add_argument("--controls", nargs="+", default=None, help="subset of controls (default: all)")
p_iso.add_argument("--max-rounds", type=int, default=20, help="round budget per run")
p_iso.add_argument("--workspace", type=str, default="outputs", help="workspace root (run dirs + reports)")

args = parser.parse_args()
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Main flow

workspace = Path(args.workspace)

if args.mode == "audit":
    run_dirs = [Path(p) for p in args.run_dirs]
    w = provider_from_env()

    # split by case_run: all windows of one run on the same side (§10)
    run_ids = [d.name for d in run_dirs]
    split = split_runs(run_ids, test_fraction=0.5, seed=12345)
    side_of = {rid: "tuning" for rid in split["tuning"]}
    side_of.update({rid: "test" for rid in split["test"]})

    # audit job assembly: (view, run_dir)
    audit_jobs = [(view, d) for view in args.views for d in run_dirs]
    acc: dict[str, dict[str, dict[str, list]]] = {
        view: {"tuning": {"j": [], "l": []}, "test": {"j": [], "l": []}}
        for view in args.views
    }
    job_idx = 0

    # Go!
    while True:
        last_job = job_idx == len(audit_jobs)

        # termination
        if last_job:
            break

        # -------------------------------------------------------------------------
        # single audit job: all windows of one run under one view
        view, d = audit_jobs[job_idx]
        job_idx += 1
        r = audit_run(d, view, w)
        side = side_of[d.name]
        acc[view][side]["j"].extend(r["judged"])
        acc[view][side]["l"].extend([wl["label"] for wl in r["labels"]])
        print(
            f"audit {job_idx:03d}/{len(audit_jobs):03d} | {view}/{d.name} "
            f"({side}) | {len(r['judged'])} windows"
        )
        # -------------------------------------------------------------------------

    # report: TPR/FPR per view and per split (tuning vs test)
    ev: dict[str, Any] = {"split": split, "views": {}}
    for view in args.views:
        ev["views"][view] = {
            "tuning": tpr_fpr(acc[view]["tuning"]["j"], acc[view]["tuning"]["l"]),
            "test": tpr_fpr(acc[view]["test"]["j"], acc[view]["test"]["l"]),
        }
    out = workspace / "reports" / "exp_d_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ev, indent=2, ensure_ascii=False, default=str))
    for view, splits in ev["views"].items():
        for split_name, r in splits.items():
            print(f"{view}/{split_name}: TPR={r['tpr']} FPR={r['fpr']} (TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']})")
    print(f"report: {out}")
else:
    provider = provider_from_env()
    inst = generate(args.case, args.seed)

    # job assembly: baseline + each isolation control (§9/§10)
    jobs: list[tuple[str, CaseRunConfig]] = [
        (
            "baseline",
            CaseRunConfig(
                case_run_id=f"isoD-base-{args.method}-{args.case}-s{args.seed}",
                case_id=args.case, seed=args.seed, method=args.method,
                max_rounds=args.max_rounds,
            ),
        )
    ]
    for name in args.controls or list(ISOLATION_CONTROLS):
        spec = ISOLATION_CONTROLS[name]
        jobs.append(
            (
                name,
                CaseRunConfig(
                    case_run_id=f"isoD-{name}-{args.method}-{args.case}-s{args.seed}",
                    case_id=args.case, seed=args.seed, method=args.method,
                    policy_mode=spec["policy_mode"],
                    writable_subset=spec["writable_subset"],
                    max_rounds=args.max_rounds,
                ),
            )
        )

    comparison: dict[str, Any] = {"controls": {}}
    job_idx = 0

    # Go!
    while True:
        last_job = job_idx == len(jobs)

        # termination
        if last_job:
            break

        # -------------------------------------------------------------------------
        # single run: baseline or one isolation control
        name, config = jobs[job_idx]
        job_idx += 1
        mlog = MeasurementLog(method=config.method, case_run_id=config.case_run_id)
        run = build_run(
            config, inst.user_prompts, provider, workspace / "runs", mlog
        )
        run.run()
        run.run_dir.mkdir(parents=True, exist_ok=True)
        meas = mlog.to_dict()
        (run.run_dir / "measurement.json").write_text(
            json.dumps(meas, indent=2, ensure_ascii=False, default=str)
        )
        m = compute_metrics(meas)
        if name == "baseline":
            comparison["baseline"] = m
        else:
            # honest no-received-data handling (§9): throughput zero,
            # latency/BER censored or N/A — never fabricated zeros
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
        print(
            f"run {job_idx:03d}/{len(jobs):03d} | {config.case_run_id} | "
            f"{run.round_no} rounds"
        )
        # -------------------------------------------------------------------------

    # report: seven metrics before/after each control (no new metric, §9)
    out = workspace / "reports" / "exp_d_isolation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comparison, indent=2, ensure_ascii=False, default=str))
    base_bits = comparison["baseline"]["effective_throughput"]["bidirectional_bits"]
    print(f"baseline delivered bits: {base_bits}")
    for name, m in comparison["controls"].items():
        bits = m["effective_throughput"]["bidirectional_bits"]
        print(f"{name}: delivered bits {bits} | BER {'N/A' if m['bit_error_rate']['ber_total'] is None else m['bit_error_rate']['ber_total']}")
    print(f"report: {out}")