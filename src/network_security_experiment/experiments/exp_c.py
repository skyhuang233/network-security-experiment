"""Experiment C: lifecycle, persistence and interference (§9, §8.6).

Measures how long a published message remains recoverable from shared storage
without refreshing or retransmission.  Delayed reads Δ ∈ {0, 1, 10, 60, 300}
seconds; each condition uses an INDEPENDENT instance with the same write (a
fresh sender publishes the known payload once, then stops).  The receiver
decodes with a FRESH decoder context (never having seen the message) —
otherwise we would measure agent memory, not storage retention (§8.6).

M3 dual expiry semantics (§12):
- passive expiry (TTL reached, no cleanup): `key in c` is already False →
  logically unrecoverable; physical rows remain on disk (counted under
  Resource Overhead, NOT Persistence);
- active cleanup (`c.expire()` = "统一资源清理" condition): rows physically
  removed → physically unrecoverable.
TTL is None for the A-group comparison runs (persistent, symmetric with
M1/M2 files/dirs which do not auto-vanish, §12) and swept only here.

Background-storage interference ("后台存储活动") is a condition, not a
metric: an unrelated writer thread hammers the same backend while the delay
elapses.

Reported as a delay-recoverability curve per (method, condition): all-
recovered conditions report "recoverable for at least the tested Δ", never
"forever" (§8.6).
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from ..channels.m1_file import M1FileContentChannel
from ..channels.m2_dirname import M2DirNameChannel
from ..channels.m3_cache import M3CacheStateChannel
from ..harness.job import CaseRunConfig, build_policy, build_views
from ..measurement.recorder import MeasurementLog

ADAPTERS = {
    "M1": M1FileContentChannel,
    "M2": M2DirNameChannel,
    "M3": M3CacheStateChannel,
}

DELAYS_SECONDS = [0, 1, 10, 60, 300]


def _fresh_channel(method: str, run_dir: Path, config: CaseRunConfig):
    shutil.rmtree(run_dir, ignore_errors=True)
    mlog = MeasurementLog(method=method, case_run_id=config.case_run_id)
    adapter = ADAPTERS[method](
        root=str(run_dir / "channel"),
        measurement=mlog,
        policy=build_policy(config),
        num_agents=2,
    )
    views = build_views(adapter, config)
    return adapter, views, mlog


def _background_writer(stop: threading.Event, run_dir: Path) -> None:
    """Unrelated background storage activity on the same backend."""

    def work():
        i = 0
        noise = run_dir / "noise"
        noise.mkdir(parents=True, exist_ok=True)
        while not stop.is_set():
            p = noise / f"n{i % 50}.tmp"
            try:
                p.write_bytes(b"\x00" * 16)
                if i % 10 == 0:
                    for j in range(50):
                        q = noise / f"n{j}.tmp"
                        if q.exists():
                            q.unlink()
            except OSError:
                pass
            i += 1
            time.sleep(0.01)

    t = threading.Thread(target=work, daemon=True)
    t.start()


def m3_single_publication_payload_len(key_set_size: int = 128) -> int:
    """Max payload bytes recoverable by ONE delayed read on M3 (§8.6).

    M3 publishes one segment per round; after the sender stops refreshing,
    only the LAST publication remains on the key set.  A fresh receiver can
    therefore recover, in a single read, only a message whose whole stream
    fits one publication:  4 B stream header + payload must fit
    (K - 16 reserved) bits.  Larger payloads are structurally multi-round on
    M3 (§6 note: "达同样 payload 所需键数/轮数结构性偏多") — a delayed
    single read losing them is a carrier property, not a defect.
    """
    p_bits = key_set_size - 18
    return max(1, (p_bits - 32) // 8)


def persistence_point(
    method: str,
    delay_seconds: float,
    run_dir: Path,
    ttl_seconds: float | None = None,
    active_cleanup_at: float | None = None,
    background_activity: bool = False,
    key_set_size: int = 128,
    payload: bytes | None = None,
) -> dict[str, Any]:
    """One (method, Δ, condition) persistence measurement."""
    import secrets

    if payload is None:
        if method == "M3":
            payload = secrets.token_bytes(
                m3_single_publication_payload_len(key_set_size)
            )
        else:
            payload = secrets.token_bytes(32)
    config = CaseRunConfig(
        case_run_id=f"persist-{method}-d{delay_seconds}-ttl{ttl_seconds}",
        case_id="persistence",
        seed=0,
        method=method,
        ttl_seconds=ttl_seconds,
        key_set_size=key_set_size,
    )
    # §14 run discipline: clear shared state before the run
    shutil.rmtree(run_dir, ignore_errors=True)
    mlog = MeasurementLog(method=method, case_run_id=config.case_run_id)
    adapter = ADAPTERS[method](
        root=str(run_dir / "channel"),
        measurement=mlog,
        policy=build_policy(config),
        num_agents=2,
    )
    views = build_views(adapter, config)

    stop = threading.Event()
    if background_activity:
        _background_writer(stop, run_dir)

    # publish the FULL message, then stop refreshing (§8.6)
    adapter.set_round(1)
    adapter.send_payload(0, payload, view=views[0])
    if method == "M3":
        round_ = 1
        # advance publications until every in-flight segment is on the wire
        while any(adapter._inflight[0]) and round_ < 500:
            adapter.set_round(round_)
            adapter.publish_round_end()
            round_ += 1
    else:
        adapter.publish_round_end()

    time.sleep(delay_seconds)

    if active_cleanup_at is not None and method == "M3":
        cache = adapter._cache(0)
        removed = cache.expire()  # active cleanup (§12)
        mlog.record_event({"event": "active_cleanup", "removed": removed})

    # FRESH receiver context: a new adapter (empty read state) sharing the
    # same live backend reads for the first time — measures storage
    # retention, not agent memory (§8.6)
    rx_adapter = ADAPTERS[method](
        root=str(run_dir / "channel"),
        measurement=MeasurementLog(method=method, case_run_id=config.case_run_id + "-rx"),
        policy=build_policy(config),
        num_agents=2,
    )
    rx_views = build_views(rx_adapter, config)
    rx_adapter.set_round(2)
    msgs = rx_adapter.receive(1, view=rx_views[1])
    recovered = any(m.sender == 0 and m.data == payload for m in msgs)

    stop.set()
    return {
        "method": method,
        "delay_seconds": delay_seconds,
        "ttl_seconds": ttl_seconds,
        "active_cleanup": active_cleanup_at is not None,
        "background_activity": background_activity,
        "key_set_size": key_set_size,
        "logically_recovered": recovered,
        "payload_bytes": len(payload),
        "storage_ops": dict(mlog.overhead.storage_ops),
    }


def persistence_curve(
    method: str,
    delays: list[float] | None = None,
    ttl_seconds: float | None = None,
    active_cleanup_at: float | None = None,
    background_activity: bool = False,
    key_set_size: int = 128,
    workspace: Path | None = None,
) -> list[dict[str, Any]]:
    """One independent instance per delay condition (§8.6)."""
    delays = delays if delays is not None else DELAYS_SECONDS
    out = []
    for d in delays:
        run_dir = (
            workspace or Path("outputs/exp_c")
        ) / f"{method}-ttl{ttl_seconds}-cleanup{active_cleanup_at}-bg{background_activity}-d{d}"
        out.append(
            persistence_point(
                method=method,
                delay_seconds=d,
                run_dir=run_dir,
                ttl_seconds=ttl_seconds,
                active_cleanup_at=active_cleanup_at,
                background_activity=background_activity,
                key_set_size=key_set_size,
            )
        )
    return out


def summarize_curve(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Report format per §8.6: delay-recoverability curve; full recovery is
    reported as "recoverable for at least the tested max delay", never
    "forever"."""
    max_all = max((p["delay_seconds"] for p in points), default=0)
    all_recovered = all(p["logically_recovered"] for p in points)
    return {
        "curve": [
            {"delay_seconds": p["delay_seconds"], "recovered": p["logically_recovered"]}
            for p in points
        ],
        "summary": (
            f"recoverable for at least the tested {max_all} s"
            if all_recovered
            else f"recoverability lost before {max_all} s under this condition"
        ),
    }