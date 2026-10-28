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

Nanochat-style script; run from the repo root (package installed via `uv sync`):

    python -m tasks.exp_b
    python -m tasks.exp_b --methods M1 --ber-threshold 0.005
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
from pathlib import Path
from typing import Any

from config.case_run import CaseRunConfig
from config.environment import save_config
from src.base import MeasurementLog, build_views, mono_now
from src.m1_file import M1FileContentChannel
from src.m2_dirname import M2DirNameChannel
from src.m3_cache import M3CacheStateChannel

# -----------------------------------------------------------------------------
# CLI arguments

parser = argparse.ArgumentParser(description="Experiment B: capacity calibration (§9, §8.1)")
parser.add_argument("--workspace", type=str, default="outputs", help="workspace root (run dirs + reports)")
parser.add_argument("--methods", nargs="+", default=["M1", "M2", "M3"], help="storage carriers under study")
parser.add_argument("--ber-threshold", type=float, default=float(os.environ.get("NSE_BER_THRESHOLD", "0.01")),
                    help="BER threshold for achievable capacity (default: NSE_BER_THRESHOLD or 0.01)")
parser.add_argument("--max-rounds", type=int, default=int(os.environ.get("NSE_MAX_ROUNDS", "100")),
                    help="round budget per calibration episode (default: NSE_MAX_ROUNDS or 100)")
args = parser.parse_args()
# -----------------------------------------------------------------------------


ADAPTERS = {
    "M1": M1FileContentChannel,
    "M2": M2DirNameChannel,
    "M3": M3CacheStateChannel,
}

M3_K_SWEEP = [16, 32, 64, 128]
PAYLOAD_GRADIENT_BYTES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
BER_THRESHOLD_DEFAULT = 0.01

# CalibrationPoint + achievable_capacity live in src/metrics.py (§8.1) so
# they are importable (and testable) without executing this module-level
# main flow
from src.metrics import CalibrationPoint, achievable_capacity  # noqa: E402


# -----------------------------------------------------------------------------
# Main flow

# sweep job assembly: (method, key-set size K, offered payload bytes)
sweep: list[tuple[str, int, int]] = []
for method in args.methods:
    k_values = M3_K_SWEEP if method == "M3" else [128]
    for K in k_values:
        for size in PAYLOAD_GRADIENT_BYTES:
            sweep.append((method, K, size))

workspace = Path(args.workspace)
EPISODES = 3
points: list[CalibrationPoint] = []
job_idx = 0

# Go!
while True:
    last_job = job_idx == len(sweep)

    # termination
    if last_job:
        break

    # -------------------------------------------------------------------------
    # single calibration job
    method, K, offered_bytes = sweep[job_idx]
    job_idx += 1
    run_id = f"calib-{method}-K{K}-S{offered_bytes}"
    run_dir = workspace / "exp_b" / run_id

    if method == "M3" and K <= 16:
        note = "K <= 16 reserved header keys: structurally infeasible"
        points.append(
            CalibrationPoint(
                method=method, key_set_size=K, offered_bytes=offered_bytes,
                delivered=False, ber=None, latency_seconds=None,
                latency_rounds=None, bits_per_second=None,
                bits_per_round=None, storage_ops=0, note=note,
            )
        )
        print(f"calib {job_idx:03d}/{len(sweep):03d} | {run_id} | {note}")
        continue

    # repeatable-rate episode driver: each episode, the sender publishes one
    # payload; the receiver reads EVERY round (fire-and-forget: a missed read
    # loses that generation); the episode ends when the payload is correctly
    # delivered or `max_rounds` is hit.  The reported rate is total delivered
    # bits over total occupied rounds and total wall time across all episodes
    # — the *repeatable* rate (§8.1), including the per-episode round-trip.
    config = CaseRunConfig(
        case_run_id=run_id,
        case_id="calibration",
        seed=0,
        method=method,
        key_set_size=K,
    )
    # §14 run discipline: shared state cleared before every run
    shutil.rmtree(run_dir, ignore_errors=True)
    try:
        mlog = MeasurementLog(method=method, case_run_id=config.case_run_id)
        adapter = ADAPTERS[method](
            root=str(run_dir / "channel"),
            measurement=mlog,
            config=config,
        )
        views = build_views(adapter, config)

        mlog.mark_run_start()
        round_ = 0
        total_delivered_bits = 0
        occupied_rounds = 0
        episodes_delivered = 0
        first_latency_rounds = None
        first_latency_seconds = None
        for ep in range(EPISODES):
            payload = secrets.token_bytes(offered_bytes)
            round_ += 1
            adapter.set_round(round_)
            chunk_count_before = len(mlog.chunks)
            adapter.send_payload(0, payload, view=views[0])
            ep_send_round = round_
            chunks = list(mlog.chunks.values())
            ep_send_time = chunks[chunk_count_before].send_time
            delivered = False
            for r in range(round_, round_ + args.max_rounds):
                adapter.set_round(r)
                round_ = r
                t_rx = mono_now()
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
    except ValueError as e:
        points.append(
            CalibrationPoint(
                method=method, key_set_size=K, offered_bytes=offered_bytes,
                delivered=False, ber=None, latency_seconds=None,
                latency_rounds=None, bits_per_second=None,
                bits_per_round=None, storage_ops=0, note=str(e),
            )
        )
        print(f"calib {job_idx:03d}/{len(sweep):03d} | {run_id} | {e}")
        continue

    # BER over all comparable delivery attempts of this job
    chunks = list(mlog.chunks.values())
    ber = None
    comparable = [a for c in chunks for a in c.attempts if a.get("comparable")]
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
    all_delivered = episodes_delivered == EPISODES
    points.append(
        CalibrationPoint(
            method=method,
            key_set_size=K,
            offered_bytes=offered_bytes,
            delivered=all_delivered,
            ber=ber,
            latency_seconds=first_latency_seconds,
            latency_rounds=first_latency_rounds,
            bits_per_second=(
                total_delivered_bits / window
                if (window and total_delivered_bits)
                else None
            ),
            bits_per_round=(
                total_delivered_bits / occupied_rounds
                if occupied_rounds
                else None
            ),
            storage_ops=sum(mlog.overhead.storage_ops.values()),
            note=(
                f"only {episodes_delivered}/{EPISODES} "
                "episodes delivered in budget"
                if not all_delivered
                else ""
            ),
        )
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "measurement.json").write_text(
        json.dumps(mlog.to_dict(), indent=2, ensure_ascii=False, default=str)
    )
    print(
        f"calib {job_idx:03d}/{len(sweep):03d} | {run_id} | "
        f"delivered={all_delivered} | BER {ber}"
    )
    # -------------------------------------------------------------------------

# report
cap = achievable_capacity(points, ber_threshold=args.ber_threshold)
out = workspace / "reports" / "exp_b.json"
out.parent.mkdir(parents=True, exist_ok=True)
# §15: record the configuration this sweep ACTUALLY consumed
save_config(
    workspace / "reports" / "config.json",
    ber_threshold=args.ber_threshold,
    max_rounds=args.max_rounds,
    methods=args.methods,
    payload_gradient_bytes=PAYLOAD_GRADIENT_BYTES,
    episodes=EPISODES,
)
out.write_text(
    json.dumps(
        {
            "points": [p.__dict__ for p in points],
            "achievable_capacity": cap,
            "note": (
                "Empirically Achievable Channel Capacity under the Tested "
                "Configuration — not a Shannon capacity (§8.1).  The "
                "bits/second and bits/round optima are selected "
                "independently and may sit at different offered loads; "
                "neither is represented by the largest payload."
            ),
        },
        indent=2,
        ensure_ascii=False,
        default=str,
    )
)
print(f"experiment B: {len(points)} calibration points")
for k, v in cap.items():
    bps = v.get("capacity_bits_per_second", {})
    bpr = v.get("capacity_bits_per_round", {})
    print(
        f"  {k}: {bps.get('bits_per_second')} bits/s "
        f"(@ {bps.get('offered_bytes')} B) | "
        f"{bpr.get('bits_per_round')} bits/round (@ {bpr.get('offered_bytes')} B)"
    )
print(f"report: {out}")