"""CLI entry point: precheck, experiment A/B/C/D, controls, reports.

Examples:
    nse precheck                              # §11 dependency precheck
    nse exp-b                                 # capacity calibration (no LLM)
    nse exp-c                                 # persistence curves (no LLM)
    nse exp-a --classes I-01 I-04             # needs NSE_API_KEY/NSE_MODEL
    nse exp-d-audit                           # W audit over finished runs
    nse exp-d-isolation --case I-01 --seed 0  # isolation controls
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import OPEN_ITEMS, save_config
from .llm import provider_from_env


def _workspace() -> Path:
    import os

    ws = Path(os.environ.get("NSE_WORKSPACE", "outputs"))
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _registry_path() -> Path:
    p = _workspace() / "precheck" / "seed-registry.json"
    if not p.exists():
        raise SystemExit(
            f"seed registry not found at {p}; run `nse precheck` first"
        )
    return p


def cmd_precheck(args) -> None:
    from .tasks.precheck import run_precheck, save_registry

    report = run_precheck(target_per_class=10)
    out = _workspace() / "precheck" / "seed-registry.json"
    save_registry(report, out)
    s = report["summary"]
    print(f"classes complete: {s['classes_complete']}/30")
    print(f"structurally degenerate (excluded, recorded): {s['structurally_degenerate']}")
    print(f"valid instances: {s['total_valid']}")
    print(f"degenerate instances excluded by reseeding: {s['total_degenerate_excluded']}")
    print(f"registry saved: {out}")


def cmd_exp_a(args) -> None:
    from .experiments.exp_a import run_batch, run_negative_control

    registry = _registry_path()
    provider = provider_from_env()

    def factory():
        return provider

    runs = run_batch(
        factory, _workspace() / "runs", registry,
        methods=args.methods, classes=args.classes or None,
        seeds_per_class=args.seeds_per_class, max_rounds=args.max_rounds,
    )
    neg = run_negative_control(
        factory, _workspace() / "runs", registry,
        classes=args.classes or None, seeds_per_class=args.seeds_per_class,
        max_rounds=args.max_rounds,
    )
    out = _workspace() / "reports" / "exp_a.json"
    save_config(_workspace() / "reports" / "config.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"runs": runs, "negative_control": neg, "open_items": OPEN_ITEMS},
            indent=2, ensure_ascii=False, default=str,
        )
    )
    print(f"experiment A: {len(runs)} main runs + {len(neg)} negative control runs")
    print(f"report: {out}")


def cmd_exp_b(args) -> None:
    from .experiments.exp_b import achievable_capacity, calibrate

    points = calibrate(workspace=_workspace() / "exp_b")
    cap = achievable_capacity(points, ber_threshold=args.ber_threshold)
    out = _workspace() / "reports" / "exp_b.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "points": [p.__dict__ for p in points],
                "achievable_capacity": cap,
                "note": (
                    "Empirically Achievable Channel Capacity under the Tested "
                    "Configuration — not a Shannon capacity (§8.1)"
                ),
            },
            indent=2, ensure_ascii=False, default=str,
        )
    )
    print(f"experiment B: {len(points)} calibration points")
    for k, v in cap.items():
        print(f"  {k}: {v['max_offered_bytes']} B | {v['bits_per_round']:.0f} bits/round | BER {v['ber']}")
    print(f"report: {out}")


def cmd_exp_c(args) -> None:
    from .experiments.exp_c import persistence_curve, summarize_curve

    results = {}
    for method in args.methods:
        conditions = [
            {"name": "baseline", "kwargs": {}},
            {"name": "ttl-1s", "kwargs": {"ttl_seconds": args.ttl}},
            {"name": "ttl-1s-active-cleanup", "kwargs": {"ttl_seconds": args.ttl, "active_cleanup_at": args.ttl}},
            {"name": "background-activity", "kwargs": {"background_activity": True}},
        ]
        results[method] = {}
        for cond in conditions:
            pts = persistence_curve(
                method, delays=args.delays, workspace=_workspace() / "exp_c",
                key_set_size=args.m3_k, **cond["kwargs"],
            )
            results[method][cond["name"]] = {
                "points": pts, "summary": summarize_curve(pts),
            }
            print(f"{method} / {cond['name']}: {results[method][cond['name']]['summary']['summary']}")
    out = _workspace() / "reports" / "exp_c.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"report: {out}")


def cmd_exp_d_audit(args) -> None:
    from .experiments.exp_d import evaluate_w

    run_dirs = [Path(p) for p in args.run_dirs]
    w = provider_from_env()
    ev = evaluate_w(run_dirs, w, views=args.views)
    out = _workspace() / "reports" / "exp_d_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ev, indent=2, ensure_ascii=False, default=str))
    for view, splits in ev["views"].items():
        for split, r in splits.items():
            print(f"{view}/{split}: TPR={r['tpr']} FPR={r['fpr']} (TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']})")
    print(f"report: {out}")


def cmd_exp_d_isolation(args) -> None:
    from .experiments.exp_d import run_isolation_comparison

    provider = provider_from_env()
    comp = run_isolation_comparison(
        lambda: provider, _workspace() / "runs",
        case_id=args.case, seed=args.seed, method=args.method,
        controls=args.controls or None, max_rounds=args.max_rounds,
    )
    out = _workspace() / "reports" / "exp_d_isolation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comp, indent=2, ensure_ascii=False, default=str))
    base_bits = comp["baseline"]["effective_throughput"]["bidirectional_bits"]
    print(f"baseline delivered bits: {base_bits}")
    for name, m in comp["controls"].items():
        bits = m["effective_throughput"]["bidirectional_bits"]
        print(f"{name}: delivered bits {bits} | BER {'N/A' if m['bit_error_rate']['ber_total'] is None else m['bit_error_rate']['ber_total']}")
    print(f"report: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="nse", description=__doc__)
    sub = parser.add_subparsers(required=True)

    sub.add_parser("precheck", help="§11 dependency precheck").set_defaults(func=cmd_precheck)

    p = sub.add_parser("exp-a", help="standard comparison (real LLM agents)")
    p.add_argument("--methods", nargs="+", default=["M1", "M2", "M3"])
    p.add_argument("--classes", nargs="+", default=None)
    p.add_argument("--seeds-per-class", type=int, default=None)
    p.add_argument("--max-rounds", type=int, default=100)
    p.set_defaults(func=cmd_exp_a)

    p = sub.add_parser("exp-b", help="capacity calibration (deterministic)")
    p.add_argument("--ber-threshold", type=float, default=0.01)
    p.set_defaults(func=cmd_exp_b)

    p = sub.add_parser("exp-c", help="persistence and lifecycle (deterministic)")
    p.add_argument("--methods", nargs="+", default=["M1", "M2", "M3"])
    p.add_argument("--delays", nargs="+", type=float, default=[0, 1, 10, 60, 300])
    p.add_argument("--ttl", type=float, default=1.0)
    p.add_argument("--m3-k", type=int, default=128)
    p.set_defaults(func=cmd_exp_c)

    p = sub.add_parser("exp-d-audit", help="W audit over finished run dirs")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--views", nargs="+", default=["content", "extended"])
    p.set_defaults(func=cmd_exp_d_audit)

    p = sub.add_parser("exp-d-isolation", help="isolation controls comparison")
    p.add_argument("--case", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--method", default="M1")
    p.add_argument("--controls", nargs="+", default=None)
    p.add_argument("--max-rounds", type=int, default=20)
    p.set_defaults(func=cmd_exp_d_isolation)

    args = parser.parse_args()
    args.func(args)