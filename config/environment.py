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


def resolved_config() -> dict[str, Any]:
    """Collect the resolved run configuration (record with every report)."""
    return {
        "ab_model": os.environ.get("NSE_MODEL"),
        "ab_api_base": os.environ.get("NSE_API_BASE", "https://api.openai.com/v1"),
        "w_model": os.environ.get("NSE_W_MODEL") or os.environ.get("NSE_MODEL"),
        "max_rounds": int(os.environ.get("NSE_MAX_ROUNDS", "100")),
        "m3_key_set_size_default": int(os.environ.get("NSE_M3_K", "128")),
        "ber_threshold": float(os.environ.get("NSE_BER_THRESHOLD", "0.01")),
        "silo_bench_commit": silo_bench_commit(),
        "diskcache_version": diskcache_version(),
    }


def silo_bench_commit() -> str | None:
    """§15.1: record the referenced silo-bench git commit."""
    repo = Path(__file__).resolve().parents[1] / "references/silo-bench/acl26-silo-bench"
    if not (repo / ".git").exists():
        return None
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def diskcache_version() -> str | None:
    try:
        import diskcache

        return diskcache.__version__
    except Exception:
        return None


def save_config(path: Path) -> dict[str, Any]:
    cfg = resolved_config()
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