"""CLI entry point: precheck registry.

The experiments themselves are nanochat-style standalone scripts with their
own argparse; run them directly (package installed via `uv sync`):

    nse precheck                                          # §11 dependency precheck
    python -m tasks.exp_a                                 # needs NSE_API_KEY/NSE_MODEL
    python -m tasks.exp_b                                 # capacity calibration (no LLM)
    python -m tasks.exp_c                                 # persistence curves (no LLM)
    python -m tasks.exp_d audit runs/main-M1-*
    python -m tasks.exp_d isolation --case I-01 --seed 0
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
    from benchmark_generator.generator import run_precheck, save_registry

    report = run_precheck(target_per_class=10)
    out = _workspace() / "precheck" / "seed-registry.json"
    save_registry(report, out)
    s = report["summary"]
    print(f"classes complete: {s['classes_complete']}/30")
    print(f"structurally degenerate (excluded, recorded): {s['structurally_degenerate']}")
    print(f"valid instances: {s['total_valid']}")
    print(f"degenerate instances excluded by reseeding: {s['total_degenerate_excluded']}")
    print(f"registry saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="nse", description=__doc__)
    sub = parser.add_subparsers(required=True)

    sub.add_parser("precheck", help="§11 dependency precheck").set_defaults(func=cmd_precheck)

    args = parser.parse_args()
    args.func(args)