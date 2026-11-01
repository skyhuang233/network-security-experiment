"""CLI entry point: precheck registry.

The experiments themselves are nanochat-style standalone scripts with their
own argparse; run them directly (package installed via `uv sync`):

    nse precheck                                          # §11 dependency precheck
    python -m tasks.exp_a                                 # needs NSE_API_KEY/NSE_MODEL
    python -m tasks.exp_b                                 # capacity calibration (no LLM)
    python -m tasks.exp_c                                 # persistence curves (no LLM)
    python -m tasks.exp_d audit runs/main-M1-*
    python -m tasks.exp_d isolation --case I-06 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _workspace() -> Path:
    ws = Path(os.environ.get("NSE_WORKSPACE", "outputs"))
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def cmd_precheck(args) -> None:
    from benchmark_generator.generator import (
        BENCHMARK_CLASSES,
        BENCHMARK_SEEDS_PER_CLASS,
        run_precheck,
        save_registry,
    )

    # the recorded benchmark: the 6 classes x 5 seeds.  --all keeps the same
    # classes but sweeps 10 seeds each.
    report = run_precheck(
        target_per_class=10 if args.all else BENCHMARK_SEEDS_PER_CLASS,
        class_ids=BENCHMARK_CLASSES,
    )
    out = _workspace() / "precheck" / "seed-registry.json"
    save_registry(report, out)
    s = report["summary"]
    print(f"classes complete: {s['classes_complete']}/{len(report['classes'])}")
    print(f"structurally degenerate (excluded, recorded): {s['structurally_degenerate']}")
    print(f"valid instances: {s['total_valid']}")
    print(f"degenerate instances excluded by reseeding: {s['total_degenerate_excluded']}")
    print(f"registry saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="nse", description=__doc__)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("precheck", help="§11 dependency precheck")
    p.add_argument("--all", action="store_true",
                   help="sweep 10 seeds per class instead of the recorded benchmark's 5")
    p.set_defaults(func=cmd_precheck)

    args = parser.parse_args()
    args.func(args)