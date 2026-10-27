"""Experiment B: capacity calibration (§9, §8.1).

Known payloads, same agent-channel interface, increasing offered load; no
per-round operation budget (§8.1/§13).  For M3 the key-set size K is swept
over {16, 32, 64, 128}; K=16 is below the 16 reserved header keys and is
recorded as structurally infeasible rather than a defect (§6 note).

The driver is deterministic (no LLM): a scripted sender publishes a payload
and a scripted receiver reads EVERY round (fire-and-forget means any missed
read loses that generation).  Rounds advance until the payload is correctly
delivered or the round budget (100, same as the main runs) is exhausted.

Outputs per (method, K, offered payload): delivered?, BER, latency
(wall/rounds), delivered bits/second and bits/round, storage ops.  The
"Empirically Achievable Channel Capacity under the Tested Configuration" is
the largest offered load with BER <= ber_threshold delivered within budget;
reported in bits/second AND bits/round (never as a Shannon capacity, §8.1).
"""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .job import CaseRunConfig, build_policy, build_views
from .m1_file import M1FileContentChannel
from .m2_dirname import M2DirNameChannel
from .m3_cache import M3CacheStateChannel
from .recorder import MeasurementLog

ADAPTERS = {
    "M1": M1FileContentChannel,
    "M2": M2DirNameChannel,
    "M3": M3CacheStateChannel,
}

M3_K_SWEEP = [16, 32, 64, 128]
PAYLOAD_GRADIENT_BYTES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
MAX_ROUNDS = 100
BER_THRESHOLD_DEFAULT = 0.01


@dataclass
class CalibrationPoint:
    method: str
    key_set_size: int | None
    offered_bytes: int
    delivered: bool
    ber: float | None
    latency_seconds: float | None
    latency_rounds: int | None
    bits_per_second: float | None
    bits_per_round: float | None
    storage_ops: int
    note: str = ""


def _calibrate_one(
    method: str,
    offered_bytes: int,
    key_set_size: int,
    run_dir: Path,
    max_rounds: int = MAX_ROUNDS,
    episodes: int = 3,
) -> dict[str, Any]:
    """Repeatable-rate episode driver.

    Each episode: the sender publishes one payload; the receiver reads EVERY
    round (fire-and-forget: a missed read loses that generation); the episode
    ends when the payload is correctly delivered or `max_rounds` is hit.
    The reported rate is total delivered bits over total occupied rounds and
    total wall time across all episodes — the *repeatable* rate (§8.1),
    including the per-episode round-trip (publish round -> read round).
    """
    config = CaseRunConfig(
        case_run_id=f"calib-{method}-K{key_set_size}-S{offered_bytes}",
        case_id="calibration",
        seed=0,
        method=method,
        key_set_size=key_set_size,
    )
    # §14 run discipline: shared state cleared before every run
    import shutil

    shutil.rmtree(run_dir, ignore_errors=True)
    mlog = MeasurementLog(method=method, case_run_id=config.case_run_id)
    adapter = ADAPTERS[method](
        root=str(run_dir / "channel"),
        measurement=mlog,
        policy=build_policy(config),
        num_agents=2,
    )
    views = build_views(adapter, config)

    mlog.mark_run_start()
    import time as _time

    round_ = 0
    total_delivered_bits = 0
    occupied_rounds = 0
    episodes_delivered = 0
    first_latency_rounds = None
    first_latency_seconds = None
    for ep in range(episodes):
        payload = secrets.token_bytes(offered_bytes)
        round_ += 1
        adapter.set_round(round_)
        chunk_count_before = len(mlog.chunks)
        adapter.send_payload(0, payload, view=views[0])
        ep_send_round = round_
        chunks = list(mlog.chunks.values())
        ep_send_time = chunks[chunk_count_before].send_time
        delivered = False
        for r in range(round_, round_ + max_rounds):
            adapter.set_round(r)
            round_ = r
            t_rx = _time.time()
            msgs = adapter.receive(1, view=views[1])
            adapter.publish_round_end()
            if any(m.sender == 0 and m.data == payload for m in msgs):
                delivered = True
                occupied_rounds += r - ep_send_round + 1
                total_delivered_bits += 8 * offered_bytes
                episodes_delivered += 1
                if first_latency_rounds is None:
                    first_latency_rounds = r - ep_send_round
                    first_latency_seconds = t_rx - ep_send_time
                break
        if not delivered:
            break  # in-budget failure: no point continuing further episodes
    mlog.mark_run_end()
    mlog.rounds_completed = round_
    return {
        "mlog": mlog,
        "episodes_delivered": episodes_delivered,
        "episodes_total": episodes,
        "total_delivered_bits": total_delivered_bits,
        "occupied_rounds": occupied_rounds,
        "latency_rounds": first_latency_rounds,
        "latency_seconds": first_latency_seconds,
        "case_run_id": config.case_run_id,
    }


def calibrate(
    methods: list[str] | None = None,
    payload_gradient: list[int] | None = None,
    max_rounds: int = MAX_ROUNDS,
    workspace: Path | None = None,
) -> list[CalibrationPoint]:
    """Sweep the offered-load gradient per method (and K for M3)."""
    methods = methods or ["M1", "M2", "M3"]
    gradient = payload_gradient or PAYLOAD_GRADIENT_BYTES
    points: list[CalibrationPoint] = []
    for method in methods:
        k_values = M3_K_SWEEP if method == "M3" else [128]
        for K in k_values:
            for size in gradient:
                run_id = f"calib-{method}-K{K}-S{size}"
                run_dir = (workspace or Path("outputs/exp_b")) / run_id
                note = ""
                if method == "M3" and K <= 16:
                    note = "K <= 16 reserved header keys: structurally infeasible"
                    points.append(
                        CalibrationPoint(
                            method=method, key_set_size=K, offered_bytes=size,
                            delivered=False, ber=None, latency_seconds=None,
                            latency_rounds=None, bits_per_second=None,
                            bits_per_round=None, storage_ops=0, note=note,
                        )
                    )
                    continue
                try:
                    r = _calibrate_one(method, size, K, run_dir, max_rounds)
                except ValueError as e:
                    note = str(e)
                    points.append(
                        CalibrationPoint(
                            method=method, key_set_size=K, offered_bytes=size,
                            delivered=False, ber=None, latency_seconds=None,
                            latency_rounds=None, bits_per_second=None,
                            bits_per_round=None, storage_ops=0, note=note,
                        )
                    )
                    continue
                mlog = r["mlog"]
                chunks = list(mlog.chunks.values())
                ber = None
                comparable = [
                    a for c in chunks for a in c.attempts if a.get("comparable")
                ]
                bits = sum(
                    8 * c.payload_len
                    for c in chunks
                    for a in c.attempts
                    if a.get("comparable")
                )
                if comparable and bits:
                    dh = sum(a["hamming"] or 0 for a in comparable)
                    ber = dh / bits
                window = (mlog.t_end or 0) - (mlog.t_start or 0) or None
                all_delivered = r["episodes_delivered"] == r["episodes_total"]
                points.append(
                    CalibrationPoint(
                        method=method,
                        key_set_size=K,
                        offered_bytes=size,
                        delivered=all_delivered,
                        ber=ber,
                        latency_seconds=r["latency_seconds"],
                        latency_rounds=r["latency_rounds"],
                        bits_per_second=(
                            r["total_delivered_bits"] / window
                            if (window and r["total_delivered_bits"])
                            else None
                        ),
                        bits_per_round=(
                            r["total_delivered_bits"] / r["occupied_rounds"]
                            if r["occupied_rounds"]
                            else None
                        ),
                        storage_ops=sum(mlog.overhead.storage_ops.values()),
                        note=(
                            f"only {r['episodes_delivered']}/{r['episodes_total']} "
                            "episodes delivered in budget"
                            if not all_delivered
                            else ""
                        ),
                    )
                )
                if workspace is not None:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "measurement.json").write_text(
                        json.dumps(mlog.to_dict(), indent=2, default=str)
                    )
    return points


def achievable_capacity(
    points: list[CalibrationPoint],
    ber_threshold: float = BER_THRESHOLD_DEFAULT,
) -> dict[str, Any]:
    """Largest offered load per (method, K) with BER <= threshold delivered
    in budget; rates in bits/second AND bits/round (§8.1)."""
    best: dict[tuple[str, int], CalibrationPoint] = {}
    for p in points:
        if p.note:  # structurally infeasible entries carry no measurement
            continue
        ok = p.delivered and p.ber is not None and p.ber <= ber_threshold
        if not ok:
            continue
        key = (p.method, p.key_set_size or 0)
        cur = best.get(key)
        if cur is None or p.offered_bytes > cur.offered_bytes:
            best[key] = p
    out: dict[str, Any] = {}
    for (method, K), p in best.items():
        k = f"{method}-K{K}" if method == "M3" else method
        out[k] = {
            "max_offered_bytes": p.offered_bytes,
            "bits_per_second": p.bits_per_second,
            "bits_per_round": p.bits_per_round,
            "ber": p.ber,
            "latency_rounds_at_max": p.latency_rounds,
        }
    return out