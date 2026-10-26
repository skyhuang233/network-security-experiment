"""Run factory: build channel adapter + views + measurement log for a run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..channels.base import ChannelAdapter, ChannelPolicy, ChannelView
from ..channels.m1_file import M1FileContentChannel
from ..channels.m2_dirname import M2DirNameChannel
from ..channels.m3_cache import M3CacheStateChannel
from ..measurement.recorder import MeasurementLog
from .engine import CaseRun, CaseRunConfig

ADAPTERS: dict[str, type[ChannelAdapter]] = {
    "M1": M1FileContentChannel,
    "M2": M2DirNameChannel,
    "M3": M3CacheStateChannel,
}


def build_policy(config: CaseRunConfig) -> ChannelPolicy:
    return ChannelPolicy(
        mode=config.policy_mode,
        writable_subset=None,
        ttl_seconds=config.ttl_seconds,
        key_set_size=config.key_set_size,
    )


def build_views(adapter: ChannelAdapter, config: CaseRunConfig) -> dict[int, ChannelView]:
    if config.policy_mode == "separated":
        # control (a): each agent gets an independent backend; in this
        # adapter each agent can read only its OWN namespace -> receive()
        # always yields nothing cross-agent (channel cut)
        return {
            i: ChannelView(
                agent_id=i,
                adapter=adapter,
                can_read_senders=frozenset({i}),
                writes_dropped=False,
            )
            for i in range(config.num_agents)
        }
    if config.policy_mode == "receiver_readonly":
        # control (b): B (agent 1) reads everything but its writes drop
        return {
            i: ChannelView(
                agent_id=i,
                adapter=adapter,
                can_read_senders=frozenset(range(config.num_agents)),
                writes_dropped=(i == 1),
            )
            for i in range(config.num_agents)
        }
    return {
        i: ChannelView(
            agent_id=i,
            adapter=adapter,
            can_read_senders=frozenset(range(config.num_agents)),
            writes_dropped=False,
        )
        for i in range(config.num_agents)
    }


def build_run(
    config: CaseRunConfig,
    user_prompts: list[str],
    provider,
    workspace: Path,
) -> CaseRun:
    """Assemble one CaseRun with its own channel backend namespace.

    §14 run discipline: shared state and agent history are cleared before
    every run.
    """
    import shutil

    run_dir = workspace / config.case_run_id
    shutil.rmtree(run_dir, ignore_errors=True)
    root = run_dir / "channel"
    mlog = MeasurementLog(method=config.method, case_run_id=config.case_run_id)
    adapter = ADAPTERS[config.method](
        root=str(root),
        measurement=mlog,
        policy=build_policy(config),
        num_agents=config.num_agents,
    )
    views = build_views(adapter, config)
    return CaseRun(
        config=config,
        user_prompts=user_prompts,
        adapter=adapter,
        views=views,
        provider=provider,
        mlog=mlog,
        run_dir=run_dir,
    )