"""Persistence-experiment helpers (§8.6, experiment C).

- Paired payloads: pre-generated ONCE per run batch, persisted (sha256) in
  the report and in every measurement log, and REUSED byte-identically
  across delay conditions, methods and repeats — every instance is an
  "independent instance with the SAME write" (§8.6).
- Common active cleanup: physical removal of the published live objects on
  every carrier (M1 file bodies, M2 directories, M3 set keys), so cleanup
  has a per-method control instead of an M3-only `expire()`.
- Background interference writers: for M3, unrelated writes into the SAME
  cache directory under a small size limit — genuine LRU eviction pressure
  against the protocol keys; for M1/M2, concurrent noise file I/O on the
  same filesystem (plain files/dirs have no eviction analog).
"""

from __future__ import annotations

import hashlib
import shutil
import threading
import time
from pathlib import Path
from typing import Any


def m3_single_publication_payload_len(key_set_size: int = 128) -> int:
    """Max payload bytes recoverable by ONE delayed read on M3 (§8.6).

    M3 publishes one segment per round; after the sender stops refreshing,
    only the LAST publication remains on the key set.  A fresh receiver can
    therefore recover, in a single read, only a message whose whole stream
    fits one publication:  4 B stream header + payload must fit
    (K - 16 reserved) bits.  Larger payloads are structurally multi-round on
    THIS protocol implementation (§6 note) — a delayed single read losing
    them is a protocol property, recorded as such, not a defect.
    """
    p_bits = key_set_size - 18
    return max(1, (p_bits - 32) // 8)


def payload_info(payload: bytes) -> dict[str, Any]:
    """Persistable provenance of a paired payload (bytes + sha256)."""
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "hex": payload.hex(),
    }


# --------------------------------------------------------------- active cleanup


def run_active_cleanup(adapter, method: str) -> dict[str, Any]:
    """Physically remove every published live object from the backend.

    Per carrier: M1 unlinks the live file bodies, M2 removes the live
    (empty) directories, M3 deletes every currently-set protocol key.
    This is the common cleanup control; M3's dual-expiry `expire()` (which
    removes only TTL-expired rows) stays a separate method-specific
    condition in experiment C."""
    removed = 0
    if method == "M3":
        cache = adapter._cache(0)
        for sender in range(adapter.num_agents):
            for k in range(adapter.K):
                key = adapter._key(sender, k)
                if key in cache:  # existence probe only (§12)
                    cache.delete(key)
                    removed += 1
        return {"objects_removed": removed}
    if method == "M1":
        live = adapter.root / "live"
        if live.exists():
            for p in sorted(live.rglob("*")):
                if p.is_file():
                    p.unlink()
                    removed += 1
        return {"objects_removed": removed}
    if method == "M2":
        live = adapter.root / "live"
        if live.exists():
            for sender_dir in sorted(live.iterdir()):
                if not sender_dir.is_dir():
                    continue
                for p in sorted(sender_dir.iterdir()):
                    if p.is_dir():
                        shutil.rmtree(p)
                        removed += 1
        return {"objects_removed": removed}
    raise ValueError(method)


# --------------------------------------------------- background interference


def start_background_writer(
    stop: threading.Event,
    method: str,
    run_dir: Path,
    live_dir: Path | None = None,
    cache_size_limit: int = 2**20,
    value_bytes: int = 64 * 1024,
    max_writes: int | None = None,
) -> threading.Thread:
    """Unrelated background storage activity on the same backend.

    M3: an independent `diskcache.Cache` handle on the SAME cache directory
    with a small `size_limit`, continuously writing large unrelated values
    — once the directory exceeds the limit, diskcache LRU-evicts the older
    protocol keys: real eviction pressure, not just concurrent I/O.

    M1/M2: concurrent noise-file write/delete churn on the same filesystem
    (plain files/dirs have no eviction mechanism; the honest analog of
    "background activity" for those carriers is concurrent I/O).
    """
    if method == "M3":
        import diskcache

        if live_dir is None:
            raise ValueError("M3 background writer needs the cache live_dir")
        cache = diskcache.Cache(str(live_dir), size_limit=cache_size_limit)

        def work() -> None:
            i = 0
            while not stop.is_set() and (max_writes is None or i < max_writes):
                try:
                    cache.set(f"bg_noise_{i % 64}", b"\x00" * value_bytes)
                except OSError:
                    pass
                i += 1
                time.sleep(0.01)
    else:

        def work() -> None:
            noise = Path(run_dir) / "noise"
            noise.mkdir(parents=True, exist_ok=True)
            i = 0
            while not stop.is_set() and (max_writes is None or i < max_writes):
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
    return t


# ---------------------------------------------------------------- curve


def summarize_curve(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Report format per §8.6: delay-recoverability curve estimated over
    repeats; full recovery is reported as "recoverable for at least the
    tested max delay", never "forever"."""
    max_all = max((p["delay_seconds"] for p in points), default=0)
    all_recovered = bool(points) and all(p["logically_recovered"] for p in points)
    return {
        "curve": [
            {
                "delay_seconds": p["delay_seconds"],
                "recovered_count": p["recovered_count"],
                "repeats": p["repeats"],
                "recovered_fraction": p["recovered_fraction"],
            }
            for p in points
        ],
        "summary": (
            f"recoverable for at least the tested {max_all} s "
            f"(all repeats of every delay point recovered)"
            if all_recovered
            else f"recoverability lost before {max_all} s under this condition"
        ),
    }