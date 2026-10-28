"""Per-run experiment configuration (was job.CaseRunConfig).

Pure data: no imports from src, so config stays dependency-free and can be
consumed by any layer (exp scripts, src.base, tasks.case_run).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CaseRunConfig:
    case_run_id: str
    case_id: str
    seed: int
    method: str  # M1 | M2 | M3
    num_agents: int = 2
    max_rounds: int = 100
    policy_mode: str = "shared"
    writable_subset: tuple[str, ...] | None = None  # control (c)
    key_set_size: int = 128  # M3
    ttl_seconds: float | None = None  # M3, experiment C
    cache_size_limit: int = 2**30  # M3 diskcache size_limit in bytes
    provider_name: str = "openai"  # or "scripted" (dev-only)
    direction_label: str | None = None  # precheck stratification label