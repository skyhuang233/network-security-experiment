"""Experiment C: lifecycle, persistence and interference (§9, §8.6).

Measures how long a published message remains recoverable from shared storage
without refreshing or retransmission.  Delayed reads Δ ∈ {0, 1, 10, 60, 300}
seconds; each condition uses INDEPENDENT instances with the SAME write: the
payload bytes are pre-generated once per batch, persisted (sha256) in the
report and in every measurement log, and reused byte-identically across
delays, methods and repeats (§8.6 pairing requirement).

Paired payload sizes (§8.6):
- COMMON conditions run every carrier at the SAME payload bytes, sized to
  what M3 can express in ONE publication (`(K - 18 - 32) / 8` bytes) — the
  common conditions therefore carry literally the same information on all
  three backends and are directly comparable.
- The per-method NATIVE size (M1/M2 at --payload-bytes, M3 at its
  single-publication maximum) is a SEPARATE condition (native-size), never
  silently mixed into the common comparison: M3's single-publication bound
  is a property of this protocol implementation, so its native condition
  only supports persistence observations at its own payload size.

The receiver decodes with a FRESH decoder context (never having seen the
message) — otherwise we would measure agent memory, not storage retention
(§8.6).  The fresh receiver's measurement log is seeded with the send-side
chunk records and ground truth (evaluation-only, never shown to any agent),
so the delayed read yields derivable BER / latency / delivered-bit metrics.

M3 dual expiry semantics (§12) — method-specific conditions:
- passive expiry (TTL reached, no cleanup): `key in c` is already False →
  logically unrecoverable; physical rows remain on disk (counted under
  Resource Overhead, NOT Persistence);
- active cleanup after TTL (`c.expire()`): rows physically removed →
  physically unrecoverable.
TTL is None for the A-group comparison runs (persistent, symmetric with
M1/M2 files/dirs which do not auto-vanish, §12) and swept only here.

Background-storage interference ("后台存储活动") is a condition, not a
metric: for M3 an unrelated writer hammers the SAME cache directory under a
small size limit — genuine LRU eviction pressure against the protocol keys;
for M1/M2 it is concurrent noise-file I/O on the same filesystem (plain
files/dirs have no eviction mechanism).  Active cleanup is a COMMON
condition: M1 live file bodies are unlinked, M2 live directories removed,
M3 set keys deleted (its TTL-based `expire()` stays a separate M3
condition).

Every (method, condition, Δ) point is estimated over --repeats independent
instances (recovered_count / repeats), not a single observation.  All-
recovered conditions report "recoverable for at least the tested Δ", never
"forever" (§8.6).

Nanochat-style script; run from the repo root (package installed via `uv sync`):

    python -m tasks.exp_c
    python -m tasks.exp_c --methods M3 --delays 0 1 10 --repeats 1
"""

from __future__ import annotations

import argparse
import json
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from config.case_run import CaseRunConfig
from config.environment import save_config
from src.base import MeasurementLog, build_views
from src.m1_file import M1FileContentChannel
from src.m2_dirname import M2DirNameChannel
from src.m3_cache import M3CacheStateChannel
from src.metrics import compute_metrics
from src.persistence import (
    m3_single_publication_payload_len,
    payload_info,
    run_active_cleanup,
    start_background_writer,
    summarize_curve,
)

# -----------------------------------------------------------------------------
# CLI arguments

parser = argparse.ArgumentParser(description="Experiment C: lifecycle, persistence and interference (§9, §8.6)")
parser.add_argument("--workspace", type=str, default="outputs", help="workspace root (run dirs + reports)")
parser.add_argument("--methods", nargs="+", default=["M1", "M2", "M3"], help="storage carriers under study")
parser.add_argument("--delays", nargs="+", type=float, default=[0, 1, 10, 60, 300], help="delay conditions Δ in seconds")
parser.add_argument("--ttl", type=float, default=1.0, help="M3 cache TTL in seconds (M3-specific expiry conditions)")
parser.add_argument("--m3-k", type=int, default=128, help="M3 key-set size K")
parser.add_argument("--payload-bytes", type=int, default=32,
                    help="native payload size for M1/M2 (the COMMON size is min(this, M3's single-publication maximum))")
parser.add_argument("--m3-bg-cache-size-limit", type=int, default=2**20,
                    help="cache size limit under M3 background-activity eviction pressure (bytes)")
parser.add_argument("--repeats", type=int, default=3,
                    help="independent instances per (method, condition, Δ) point (stable-curve estimation)")
args = parser.parse_args()
# -----------------------------------------------------------------------------


ADAPTERS = {
    "M1": M1FileContentChannel,
    "M2": M2DirNameChannel,
    "M3": M3CacheStateChannel,
}

# -----------------------------------------------------------------------------
# Paired payloads (§8.6): pre-generated ONCE, persisted, reused byte-
# identically across delays, methods (common) and repeats.  The common size
# is what M3 can express in one publication, so common conditions carry the
# same information on every carrier.

m3_native_len = m3_single_publication_payload_len(args.m3_k)
common_len = min(args.payload_bytes, m3_native_len)
PAYLOAD_COMMON = secrets.token_bytes(common_len)
native_m12 = secrets.token_bytes(args.payload_bytes)
PAYLOAD_NATIVE: dict[str, bytes] = {
    "M1": native_m12,
    "M2": native_m12,  # same bytes: the write is paired across methods too
    "M3": PAYLOAD_COMMON if m3_native_len == common_len
    else secrets.token_bytes(m3_native_len),
}
PAYLOAD_REGISTRY = {
    "common": payload_info(PAYLOAD_COMMON),
    "native": {m: payload_info(b) for m, b in PAYLOAD_NATIVE.items()},
}

# -----------------------------------------------------------------------------
# Conditions (§8.6):
#   COMMON: every carrier, paired common payload
#   M3-specific: dual-expiry semantics (cache-backend mechanisms; M1/M2
#                files/dirs have no passive-expiry analog) — listed
#                separately, never mixed into the common comparison
#   NATIVE: per-method native payload size, baseline semantics

COMMON_CONDITIONS = [
    {"name": "baseline", "kwargs": {}, "payload": "common"},
    # common active cleanup: physical removal of the published live objects
    # on every carrier (M1 unlink files / M2 remove dirs / M3 delete keys)
    {"name": "active-cleanup", "kwargs": {"active_cleanup": True}, "payload": "common"},
    # background interference: M3 same-cache eviction pressure,
    # M1/M2 concurrent filesystem noise I/O
    {"name": "background-activity", "kwargs": {"background_activity": True}, "payload": "common"},
]
M3_CONDITIONS = [
    # passive expiry: logically unrecoverable, physical rows remain (§12)
    {"name": "ttl-1s", "kwargs": {"ttl_seconds": args.ttl}, "payload": "common"},
    # passive expiry THEN active expire(): physically unrecoverable (§12)
    {"name": "ttl-1s-active-cleanup",
     "kwargs": {"ttl_seconds": args.ttl, "active_cleanup_at": args.ttl, "expire_cleanup": True},
     "payload": "common"},
]
NATIVE_CONDITION = {"name": "native-size", "kwargs": {}, "payload": "native"}

# job assembly: one INDEPENDENT instance per (method, condition, Δ, repeat)
jobs: list[tuple[str, dict, float, int]] = []
for method in args.methods:
    for cond in COMMON_CONDITIONS:
        for d in args.delays:
            for r in range(args.repeats):
                jobs.append((method, cond, d, r))
    if method == "M3":
        for cond in M3_CONDITIONS:
            for d in args.delays:
                for r in range(args.repeats):
                    jobs.append((method, cond, d, r))
    for d in args.delays:
        for r in range(args.repeats):
            jobs.append((method, NATIVE_CONDITION, d, r))

workspace = Path(args.workspace)
results: dict[str, dict[str, list[dict[str, Any]]]] = {
    m: {} for m in args.methods
}
job_idx = 0

# Go!
while True:
    last_job = job_idx == len(jobs)

    # termination
    if last_job:
        break

    # -------------------------------------------------------------------------
    # single persistence measurement (one repeat of one point)
    method, cond, delay_seconds, repeat = jobs[job_idx]
    job_idx += 1
    cond_name = cond["name"]
    ttl_seconds = cond["kwargs"].get("ttl_seconds")
    active_cleanup = cond["kwargs"].get("active_cleanup", False)
    active_cleanup_at = cond["kwargs"].get("active_cleanup_at")
    expire_cleanup = cond["kwargs"].get("expire_cleanup", False)
    background_activity = cond["kwargs"].get("background_activity", False)
    payload = PAYLOAD_COMMON if cond["payload"] == "common" else PAYLOAD_NATIVE[method]

    run_dir = workspace / "exp_c" / (
        f"{method}-{cond_name}-d{delay_seconds}-r{repeat}"
    )
    config = CaseRunConfig(
        case_run_id=f"persist-{method}-{cond_name}-d{delay_seconds}-r{repeat}",
        case_id="persistence",
        seed=0,
        method=method,
        ttl_seconds=ttl_seconds,
        key_set_size=args.m3_k,
        # eviction pressure is part of the background-activity condition:
        # the cache backend itself runs under a small size limit
        cache_size_limit=(
            args.m3_bg_cache_size_limit
            if background_activity and method == "M3"
            else 2**30
        ),
    )
    # §14 run discipline: clear shared state before the run
    shutil.rmtree(run_dir, ignore_errors=True)
    mlog = MeasurementLog(method=method, case_run_id=config.case_run_id)
    adapter = ADAPTERS[method](
        root=str(run_dir / "channel"),
        measurement=mlog,
        config=config,
    )
    views = build_views(adapter, config)
    # paired-write provenance travels inside the measurement log (§8.6)
    mlog.record_event(
        {
            "event": "payload",
            "round": 0,
            "condition": cond_name,
            "payload_kind": cond["payload"],
            **payload_info(payload),
        }
    )

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

    stop = threading.Event()
    if background_activity:
        # the writer hammers the backend WHILE THE DELAY ELAPSES (after the
        # publication completed) — the condition measures retention under
        # ongoing unrelated activity, not publication under interference
        start_background_writer(
            stop, method, run_dir,
            live_dir=adapter.live_dir if method == "M3" else None,
            cache_size_limit=args.m3_bg_cache_size_limit,
        )

    time.sleep(delay_seconds)

    if active_cleanup:
        # common active cleanup: physical removal of the live objects
        info = run_active_cleanup(adapter, method)
        mlog.record_event({"event": "active_cleanup", **info})
    elif active_cleanup_at is not None:
        if method == "M3":
            # M3 dual-expiry cleanup (§12): physically remove TTL-expired rows
            cache = adapter._cache(0)
            removed = cache.expire()
            mlog.record_event({"event": "active_cleanup", "removed": removed})
        else:
            info = run_active_cleanup(adapter, method)
            mlog.record_event({"event": "active_cleanup", **info})

    # FRESH receiver context: a new adapter (empty read state) sharing the
    # same live backend reads for the first time — measures storage
    # retention, not agent memory (§8.6)
    rx_log = MeasurementLog(method=method, case_run_id=config.case_run_id + "-rx")
    rx_adapter = ADAPTERS[method](
        root=str(run_dir / "channel"),
        measurement=rx_log,
        config=config,
    )
    rx_views = build_views(rx_adapter, config)
    # seed the fresh receiver's measurement layer with the send-side chunk
    # records and ground truth (evaluation-only; never shown to any agent)
    # so the delayed read yields comparable, derivable BER/latency metrics
    rx_adapter._truth.update(adapter._truth)
    for rec in mlog.chunks.values():
        r = rx_log.register_send(
            sender=rec.sender,
            msg_id=rec.msg_id,
            chunk_id=rec.chunk_id,
            total_chunks=rec.total_chunks,
            frame_published=rec.frame_published,
            payload_len=rec.payload_len,
        )
        r.send_round = rec.send_round
        r.send_time = rec.send_time
        r.publish_round = rec.publish_round
    rx_adapter.set_round(2)
    msgs = rx_adapter.receive(1, view=rx_views[1])
    recovered = any(m.sender == 0 and m.data == payload for m in msgs)

    rx_metrics = compute_metrics(rx_log.to_dict())
    rx_summary = {
        "delivered_bits": rx_metrics["effective_throughput"]["bidirectional_bits"],
        "ber_total": rx_metrics["bit_error_rate"]["ber_total"],
        "latency_seconds_median": next(
            (
                v["latency_seconds_median"]
                for v in rx_metrics["communication_latency"].values()
            ),
            None,
        ),
    }

    stop.set()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "measurement.json").write_text(
        json.dumps(mlog.to_dict(), indent=2, ensure_ascii=False, default=str)
    )
    (run_dir / "receiver").mkdir(parents=True, exist_ok=True)
    (run_dir / "receiver" / "measurement.json").write_text(
        json.dumps(rx_log.to_dict(), indent=2, ensure_ascii=False, default=str)
    )

    # accumulate the repeat into its (method, condition, Δ) point
    bucket = results[method].setdefault(
        f"{cond_name}", []
    )
    point_key = (
        delay_seconds,
        ttl_seconds,
        active_cleanup or active_cleanup_at is not None,
        background_activity,
        cond["payload"],
    )
    point = next(
        (p for p in bucket if p["point_key"] == point_key),
        None,
    )
    if point is None:
        point = {
            "point_key": point_key,
            "method": method,
            "condition": cond_name,
            "payload": cond["payload"],
            "payload_bytes": len(payload),
            "payload_sha256": payload_info(payload)["sha256"],
            "delay_seconds": delay_seconds,
            "ttl_seconds": ttl_seconds,
            "active_cleanup": active_cleanup or active_cleanup_at is not None,
            "expire_cleanup": expire_cleanup,
            "background_activity": background_activity,
            "key_set_size": args.m3_k,
            "cache_size_limit": config.cache_size_limit,
            "repeats": 0,
            "recovered_count": 0,
            "receiver_metrics": [],
        }
        bucket.append(point)
    point["repeats"] += 1
    point["recovered_count"] += 1 if recovered else 0
    point["receiver_metrics"].append(rx_summary)
    print(
        f"persist {job_idx:03d}/{len(jobs):03d} | {method} / {cond_name} "
        f"| Δ={delay_seconds}s | r{repeat} | recovered={recovered}"
    )
    # -------------------------------------------------------------------------

# finalize points: fraction + boolean per §8.6 (estimated over repeats)
for method in results:
    for cond_name, bucket in results[method].items():
        for p in bucket:
            p.pop("point_key", None)
            p["recovered_fraction"] = (
                p["recovered_count"] / p["repeats"] if p["repeats"] else None
            )
            p["logically_recovered"] = (
                p["recovered_count"] == p["repeats"] and p["repeats"] > 0
            )

# report: delay-recoverability curve per (method, condition)
for method in results:
    for cond_name in results[method]:
        pts = results[method][cond_name]
        results[method][cond_name] = {
            "points": pts,
            "summary": summarize_curve(pts),
        }
        print(f"{method} / {cond_name}: {results[method][cond_name]['summary']['summary']}")

out = workspace / "reports" / "exp_c.json"
out.parent.mkdir(parents=True, exist_ok=True)
# §15: record the configuration this experiment ACTUALLY consumed, plus the
# paired-payload registry (same bytes across delays/methods/repeats, §8.6)
save_config(
    workspace / "reports" / "config.json",
    max_rounds=None,
    m3_key_set_size_default=args.m3_k,
    methods=args.methods,
    delays=args.delays,
    ttl=args.ttl,
    payload_bytes=args.payload_bytes,
    repeats=args.repeats,
    m3_bg_cache_size_limit=args.m3_bg_cache_size_limit,
)
out.write_text(
    json.dumps(
        {
            "payloads": PAYLOAD_REGISTRY,
            "results": results,
        },
        indent=2,
        ensure_ascii=False,
        default=str,
    )
)
print(f"report: {out}")