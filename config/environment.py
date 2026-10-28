"""§15 experiment environment configuration.

Values come from environment variables so the code is repo-stable and the
run artifacts record the resolved configuration.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


def resolved_config(**overrides: Any) -> dict[str, Any]:
    """Collect the resolved run configuration (record with every report).

    Environment variables provide the DEFAULTS only; callers pass the values
    their runner ACTUALLY CONSUMED as keyword overrides (e.g. the argparse
    values after env-defaulting).  The record can therefore never drift from
    the executed configuration (§15): what is written is what ran."""
    cfg = {
        "ab_model": os.environ.get("NSE_MODEL"),
        "ab_api_base": os.environ.get("NSE_API_BASE", "https://api.openai.com/v1"),
        "w_model": os.environ.get("NSE_W_MODEL") or os.environ.get("NSE_MODEL"),
        "max_rounds": int(os.environ.get("NSE_MAX_ROUNDS", "100")),
        "m3_key_set_size_default": int(os.environ.get("NSE_M3_K", "128")),
        "ber_threshold": float(os.environ.get("NSE_BER_THRESHOLD", "0.01")),
        "silo_bench_commit": silo_bench_commit(),
        "silo_bench_provenance": silo_bench_provenance(),
        "diskcache_version": diskcache_version(),
    }
    cfg.update(overrides)
    return cfg


# The pinned SILO-BENCH source is NOT committed (references/ is gitignored),
# so a clean checkout cannot probe its .git.  The upstream pin is therefore
# ALSO recorded here, in committed code, and used as the fallback whenever
# the local checkout is absent — upstream provenance never depends on an
# untracked directory (§15.1).
SILO_BENCH_UPSTREAM_URL = "https://github.com/jwyjohn/acl26-silo-bench"
SILO_BENCH_PINNED_COMMIT = "e74127782ed1c42fff474249961f022c063d76f2"

_silo_bench_local_git = True  # set by silo_bench_commit() on each probe


def silo_bench_commit() -> str | None:
    """§15.1: record the referenced silo-bench git commit.

    Probes the local checkout under references/ first; if it is absent (a
    clean checkout — the directory is gitignored) or the probe fails, fall
    back to the committed SILO_BENCH_PINNED_COMMIT instead of silently
    returning None."""
    global _silo_bench_local_git
    repo = Path(__file__).resolve().parents[1] / "references/silo-bench/acl26-silo-bench"
    if (repo / ".git").exists():
        try:
            out = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo, capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            if out:
                _silo_bench_local_git = True
                return out
        except (OSError, subprocess.TimeoutExpired):
            pass
    _silo_bench_local_git = False
    return SILO_BENCH_PINNED_COMMIT


def silo_bench_provenance() -> str:
    """Where the recorded silo-bench commit came from (§15.1)."""
    return (
        f"local checkout at {SILO_BENCH_UPSTREAM_URL} (rev-parse HEAD)"
        if _silo_bench_local_git
        else f"committed pin of {SILO_BENCH_UPSTREAM_URL} (references/ is "
        "gitignored; clean checkouts use the pin)"
    )


def diskcache_version() -> str | None:
    try:
        import diskcache

        return diskcache.__version__
    except Exception:
        return None


def save_config(path: Path, **overrides: Any) -> dict[str, Any]:
    """Persist the resolved config; `overrides` carry the values the calling
    runner actually consumed (argparse results, W model binding, ...) so the
    record matches the executed run (§15)."""
    cfg = resolved_config(**overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    return cfg


OPEN_ITEMS = {
    "model_binding": (
        "§15.3: set NSE_MODEL (+ NSE_API_BASE / NSE_API_KEY) before recorded "
        "runs; the resolved version string is stamped into every report"
    ),
    "m3_k_final": (
        "§15.4: final K for M3 main runs is chosen after experiment B "
        "capacity calibration; the sweep covers {16,32,64,128}"
    ),
    "repetitions": (
        "§14: whether to increase repetitions is decided from pilot "
        "variance BEFORE formal runs, not because a group looks good"
    ),
}