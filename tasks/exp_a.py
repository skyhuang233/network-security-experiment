"""Experiment A: standard task comparison (§9) + the two controls (§9 end).

For each of the 29 valid task classes (I-03 structurally degenerate) x 10
valid seeds (precheck registry) x 3 methods (M1/M2/M3), run the full
round loop with real LLM task agents and compute the seven metrics per run.

Run order is randomized per batch; shared state and agent history are
cleared before every run (§14).

The negative control ("no cross-agent shared state") runs the SAME tasks
with policy_mode="separated": A and B get seemingly-identical but
independent storage backends, keep reading and writing allowed work
products, but have NO cross-agent channel — the real-negative source for
experiment D (§10).  It runs ALL methods under study (not just M1) so the
audit stage has per-method normal-operation background (M2 directory
operations, M3 cache operations) for the false-positive side.

The positive environment control ("authorized direct communication") uses a
direct-mailbox pseudo-channel to verify the model/task/execution framework
itself works; it is NOT one of the three storage carriers under study.

The LLM round loop (CaseRun + build_run) lives in tasks/round_loop.py and is
shared with exp_d; each script keeps its own module-level argparse + main
flow, so importing the shared runtime is safe.

Nanochat-style script; run from the repo root (package installed via `uv sync`):

    python -m tasks.exp_a --classes I-01 I-04
    python -m tasks.exp_a --seeds-per-class 2

Recorded runs use the real LLM (`--provider openai`, the default; requires
NSE_API_KEY / NSE_MODEL).  `--provider scripted` replays the dev-only
ScriptedProvider with NO API calls — smoke tests only, never usable as
recorded experiment data (§1).
"""

from __future__ import annotations

import argparse
import json
import os
import random as _random
from pathlib import Path
from typing import Any

from config.case_run import CaseRunConfig
from src import frame as F
from src.base import ChannelAdapter, MeasurementLog
from src.audit_agent import provider_from_env
from src.metrics import compute_metrics
from tasks.round_loop import ADAPTERS, build_run
from benchmark_generator.generator import generate, load_registry

# -----------------------------------------------------------------------------
# CLI arguments

parser = argparse.ArgumentParser(
    description="Experiment A: standard task comparison (§9) + controls"
)
parser.add_argument("--workspace", type=str, default="outputs", help="workspace root (run dirs + reports)")
parser.add_argument("--registry", type=str, default="", help="seed registry path (empty = <workspace>/precheck/seed-registry.json)")
parser.add_argument("--methods", nargs="+", default=["M1", "M2", "M3"], help="storage carriers under study")
parser.add_argument("--classes", nargs="+", default=None, help="restrict to these task classes (default: all valid)")
parser.add_argument("--seeds-per-class", type=int, default=None, help="cap valid seeds per class (default: all)")
parser.add_argument("--max-rounds", type=int, default=int(os.environ.get("NSE_MAX_ROUNDS", "100")), help="round budget per run (default: NSE_MAX_ROUNDS or 100)")
parser.add_argument("--order-seed", type=int, default=2026, help="seed for the randomized run order (§14)")
parser.add_argument("--no-negative-control", action="store_true", help="skip the separated-backend negative control runs")
parser.add_argument("--direct-control", action="store_true", help="also run the authorized-direct-communication positive control")
parser.add_argument("--provider", choices=["openai", "scripted"], default="openai",
                    help="LLM provider: real API (recorded runs, default) or dev-only scripted replay (smoke tests; never usable as data)")
args = parser.parse_args()
# -----------------------------------------------------------------------------


class DirectMailboxChannel(ChannelAdapter):
    """Environment-validation control: SILO-msg-style direct messaging.

    Same send_payload/receive interface, one-round visibility, but delivery
    is via a direct mailbox (no storage carrier).  Verifies that the task,
    model binding and execution framework work when communication is easy.
    Never used for the M1/M2/M3 measurements.
    """

    method = "CTRL-DIRECT"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mailboxes: dict[int, list[tuple[int, bytes]]] = {
            i: [] for i in range(self.num_agents)
        }
        self._stage_box: dict[int, list[tuple[int, bytes]]] = {
            i: [] for i in range(self.num_agents)
        }

    def _stage_frame(self, sender: int, fr: F.Frame) -> bool:
        # same one-round visibility rule as the storage channels: staged
        # this round, readable from next round
        self._stage_box[sender].append((sender, fr.pack()))
        return True

    def _promote(self, sender: int) -> None:
        if self._stage_box[sender]:
            self._mailboxes[(sender + 1) % self.num_agents].extend(
                self._stage_box[sender]
            )
            self._stage_box[sender] = []

    def _read_new_live(self, sender: int, receiver: int) -> list[F.WireFrame]:
        box = self._mailboxes[receiver]
        out = []
        while box:
            _s, raw = box.pop(0)
            wf = F.WireFrame.parse(raw)
            if wf is not None:
                out.append(wf)
        return out


def build_jobs(
    registry: dict[str, Any],
    condition: str,
    policy_mode: str,
    run_tag: str,
    methods: list[str] | None = None,
    classes: list[str] | None = None,
    seeds_per_class: int | None = None,
    max_rounds: int = 100,
    order_seed: int = 2026,
    key_set_size: int = 128,
    provider_name: str = "openai",
) -> list[tuple[CaseRunConfig, str]]:
    """Assemble one batch of (config, condition) jobs from the seed registry.

    Run order is randomized per batch (§14).  `key_set_size` and
    `provider_name` are the values the runner actually consumes (§15).
    """
    jobs: list[tuple[CaseRunConfig, str]] = []
    methods = methods or ["M1", "M2", "M3"]
    for case_id, cinfo in registry["classes"].items():
        if cinfo["structurally_degenerate"]:
            continue
        if classes and case_id not in classes:
            continue
        seeds = [v["seed"] for v in cinfo["valid"]][:seeds_per_class]
        direction = {v["seed"]: v["label"] for v in cinfo["valid"]}
        for method in methods:
            for seed in seeds:
                jobs.append(
                    (
                        CaseRunConfig(
                            case_run_id=f"{run_tag}-{method}-{case_id}-s{seed}",
                            case_id=case_id,
                            seed=seed,
                            method=method,
                            policy_mode=policy_mode,
                            max_rounds=max_rounds,
                            key_set_size=key_set_size,
                            provider_name=provider_name,
                            direction_label=direction.get(seed),
                        ),
                        condition,
                    )
                )
    # §14: randomized run order per batch
    _random.Random(order_seed).shuffle(jobs)
    return jobs


# -----------------------------------------------------------------------------
# Main flow

from config.environment import OPEN_ITEMS, save_config  # noqa: E402

workspace = Path(args.workspace)
workspace.mkdir(parents=True, exist_ok=True)
registry_path = (
    Path(args.registry)
    if args.registry
    else workspace / "precheck" / "seed-registry.json"
)
if not registry_path.exists():
    raise FileNotFoundError(
        f"seed registry not found at {registry_path}; run `nse precheck` first"
    )
registry = load_registry(registry_path)

# provider binding actually consumed by this batch (§15): the real API for
# recorded runs; the dev-only scripted replay for smoke tests (never usable
# as recorded experiment data)
if args.provider == "scripted":
    from tasks.round_loop import scripted_provider

    provider = scripted_provider()
else:
    provider = provider_from_env()

# M3's key-set size is consumed from the same env var that is recorded
# (§15.4); the runner and the report can never disagree
m3_key_set_size = int(os.environ.get("NSE_M3_K", "128"))

# adapter table: the three carriers, plus the direct-mailbox control when
# requested (registered into a local copy, never into the shared table)
adapters = dict(ADAPTERS)
if args.direct_control:
    adapters["CTRL-DIRECT"] = DirectMailboxChannel

# job assembly: main runs (M1/M2/M3, shared backends) plus the
# separated-backend negative control (§9/§10).  The negative control runs
# ALL methods under study: the audit stage (experiment D) needs per-method
# normal-operation background (M2 directory ops, M3 cache ops) for the
# false-positive side, not M1 files only.
jobs = build_jobs(
    registry,
    condition="main",
    policy_mode="shared",
    run_tag="main",
    methods=args.methods,
    classes=args.classes,
    seeds_per_class=args.seeds_per_class,
    max_rounds=args.max_rounds,
    order_seed=args.order_seed,
    key_set_size=m3_key_set_size,
    provider_name=args.provider,
)
if not args.no_negative_control:
    jobs += build_jobs(
        registry,
        condition="negative-control",
        policy_mode="separated",
        run_tag="negctrl",
        methods=args.methods,
        classes=args.classes,
        seeds_per_class=args.seeds_per_class,
        max_rounds=args.max_rounds,
        order_seed=args.order_seed,
        key_set_size=m3_key_set_size,
        provider_name=args.provider,
    )
if args.direct_control:
    jobs += build_jobs(
        registry,
        condition="direct-control",
        policy_mode="shared",
        run_tag="directctrl",
        methods=["CTRL-DIRECT"],
        classes=args.classes,
        seeds_per_class=args.seeds_per_class,
        max_rounds=args.max_rounds,
        order_seed=args.order_seed,
        key_set_size=m3_key_set_size,
        provider_name=args.provider,
    )

# Go!
job_idx = 0
runs: list[dict[str, Any]] = []
while True:
    last_job = job_idx == len(jobs)

    # write the report: at the end of the run
    if last_job:
        break

    # -------------------------------------------------------------------------
    # single run
    config, condition = jobs[job_idx]
    job_idx += 1
    inst = generate(config.case_id, config.seed)
    mlog = MeasurementLog(method=config.method, case_run_id=config.case_run_id)
    run = build_run(
        config, inst.user_prompts, provider, workspace / "runs", mlog,
        adapters=adapters,
    )
    result = run.run()
    run.run_dir.mkdir(parents=True, exist_ok=True)
    meas = mlog.to_dict()
    (run.run_dir / "measurement.json").write_text(
        json.dumps(meas, indent=2, ensure_ascii=False, default=str)
    )
    metrics = compute_metrics(meas)
    runs.append(
        {
            "case_run_id": config.case_run_id,
            "case_id": config.case_id,
            "seed": config.seed,
            "method": config.method,
            "condition": condition,
            "policy_mode": config.policy_mode,
            "direction_label": config.direction_label,
            "key_set_size": config.key_set_size,
            "max_rounds": config.max_rounds,
            "provider": config.provider_name,
            "rounds": result["rounds"],
            "wall_seconds": result["wall_seconds"],
            "submissions": result["submissions"],
            "metrics": metrics,
        }
    )
    print(
        f"run {job_idx:03d}/{len(jobs):03d} | {config.case_run_id} | "
        f"{result['rounds']} rounds | {result['wall_seconds']:.1f}s"
    )
    # -------------------------------------------------------------------------

# report
runs_main = [r for r in runs if r["condition"] == "main"]
neg = [r for r in runs if r["condition"] == "negative-control"]
direct = [r for r in runs if r["condition"] == "direct-control"]
out = workspace / "reports" / "exp_a.json"
# §15: record the configuration this batch ACTUALLY consumed (argparse values
# after env-defaulting), not a second, parallel env reading
save_config(
    workspace / "reports" / "config.json",
    max_rounds=args.max_rounds,
    m3_key_set_size_default=m3_key_set_size,
    methods=args.methods,
    provider=args.provider,
)
out.parent.mkdir(parents=True, exist_ok=True)
report: dict[str, Any] = {
    "runs": runs_main,
    "negative_control": neg,
    "direct_control": direct,
    "provider": args.provider,
    "open_items": OPEN_ITEMS,
}
if args.provider == "scripted":
    report["warning"] = (
        "scripted-provider runs are dev-only smoke tests and are never "
        "usable as recorded experiment data (§1)"
    )
out.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
        default=str,
    )
)
print(f"experiment A: {len(runs_main)} main runs + {len(neg)} negative control runs")
print(f"report: {out}")