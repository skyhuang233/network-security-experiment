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
experiment D (§10).

The positive environment control ("authorized direct communication") uses a
direct-mailbox pseudo-channel to verify the model/task/execution framework
itself works; it is NOT one of the three storage carriers under study.
"""

from __future__ import annotations

import json
import random as _random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..channels.base import ChannelAdapter, ChannelPolicy, ChannelView
from ..evaluation.metrics import compute_metrics
from ..harness.engine import CaseRun, CaseRunConfig
from ..harness.runner import build_run
from ..protocol import frame as F
from ..tasks.precheck import load_registry
from ..tasks.registry import generate


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


def run_batch(
    provider_factory,
    workspace: Path,
    registry_path: Path,
    methods: list[str] | None = None,
    policy_mode: str = "shared",
    condition: str = "main",
    classes: list[str] | None = None,
    seeds_per_class: int | None = None,
    max_rounds: int = 100,
    order_seed: int = 2026,
    run_tag: str | None = None,
    provider_name: str = "openai",
) -> list[dict[str, Any]]:
    """Run a batch of case runs and return per-run metric summaries.

    provider_factory() -> LLMProvider is called once per run (fresh agent
    bindings per run; agents share one fixed model version, §13).
    """
    registry = load_registry(registry_path)
    jobs: list[CaseRunConfig] = []
    methods = methods or ["M1", "M2", "M3"]
    tag = run_tag or f"{condition}-{policy_mode}"
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
                    CaseRunConfig(
                        case_run_id=f"{tag}-{method}-{case_id}-s{seed}",
                        case_id=case_id,
                        seed=seed,
                        method=method,
                        policy_mode=policy_mode,
                        max_rounds=max_rounds,
                        provider_name=provider_name,
                        direction_label=direction.get(seed),
                    )
                )
    # §14: randomized run order per batch
    _random.Random(order_seed).shuffle(jobs)

    out: list[dict[str, Any]] = []
    for config in jobs:
        inst = generate(config.case_id, config.seed)
        run = build_run(config, inst.user_prompts, provider_factory(), workspace)
        result = run.run()
        meas = json.loads(
            (workspace / config.case_run_id / "measurement.json").read_text()
        )
        metrics = compute_metrics(meas)
        out.append(
            {
                "case_run_id": config.case_run_id,
                "case_id": config.case_id,
                "seed": config.seed,
                "method": config.method,
                "condition": condition,
                "policy_mode": policy_mode,
                "direction_label": config.direction_label,
                "rounds": result["rounds"],
                "wall_seconds": result["wall_seconds"],
                "submissions": result["submissions"],
                "metrics": metrics,
            }
        )
    return out


def run_negative_control(
    provider_factory, workspace: Path, registry_path: Path, **kwargs
) -> list[dict[str, Any]]:
    """无跨 agent 共享状态对照 (§9/§10): separated backends -> no channel."""
    return run_batch(
        provider_factory,
        workspace,
        registry_path,
        methods=["M1"],  # carrier shape irrelevant; one method suffices
        policy_mode="separated",
        condition="negative-control",
        run_tag="negctrl",
        **kwargs,
    )


def run_direct_control(
    provider_factory, workspace: Path, registry_path: Path, **kwargs
) -> list[dict[str, Any]]:
    """授权直接通信对照 (§9): verifies model/task/framework sanity."""

    # run_batch wires the adapter by method name; register the control
    import network_security_experiment.harness.runner as runner_mod

    original = dict(runner_mod.ADAPTERS)
    runner_mod.ADAPTERS["CTRL-DIRECT"] = DirectMailboxChannel
    try:
        return run_batch(
            provider_factory,
            workspace,
            registry_path,
            methods=["CTRL-DIRECT"],
            condition="direct-control",
            run_tag="directctrl",
            **kwargs,
        )
    finally:
        runner_mod.ADAPTERS.clear()
        runner_mod.ADAPTERS.update(original)