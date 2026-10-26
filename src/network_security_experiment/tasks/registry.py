"""Task registry: 30 SILO-BENCH classes, seed-parameterized generation (§14).

`generate(case_id, seed, num_agents=2)` produces one independent task
instance.  The seed -> instance mapping is stored in the run metadata for
reproduction.  All three methods (M1/M2/M3) share the same seed set and the
same private shards.
"""

from __future__ import annotations

import random
from typing import Any

from .paradigm_i import PARADIGM_I
from .paradigm_ii import PARADIGM_II
from .paradigm_iii import PARADIGM_III

ALL_CLASSES: dict[str, Any] = {**PARADIGM_I, **PARADIGM_II, **PARADIGM_III}

NUM_AGENTS = 2  # fixed by §14: all 30 classes at n=2


def _safe_format(template: str, **kwargs) -> str:
    """Literal-replace only known placeholders; keep other braces untouched
    (same semantics as SILO's `_safe_prompt_format`)."""
    out = template
    for k, v in kwargs.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def build_user_prompt(case: dict[str, Any], agent_id: int, num_agents: int) -> str:
    return _safe_format(
        case["description"],
        agent_id=agent_id,
        num_agents=num_agents,
        max_id=num_agents - 1,
        input_shard=case["input_shards"][agent_id],
    )


class TaskInstance:
    """One (case_class, seed) instance with per-agent prompts and shards."""

    def __init__(self, case: dict[str, Any], seed: int, num_agents: int) -> None:
        self.case_id = case["case_id"]
        self.case_name = case["case_name"]
        self.paradigm = case["paradigm"]
        self.seed = seed
        self.num_agents = num_agents
        self.is_segmented = case["is_segmented"]
        self.input_shards = case["input_shards"]
        self.expected_per_agent = case["expected_per_agent"]
        self.user_prompts = [
            build_user_prompt(case, i, num_agents) for i in range(num_agents)
        ]
        self.raw = case  # description template + unsharded data for oracles

    @property
    def instance_id(self) -> str:
        return f"{self.case_id}_seed{self.seed:04d}_n{self.num_agents}"


def generate(case_id: str, seed: int, num_agents: int = NUM_AGENTS) -> TaskInstance:
    if case_id not in ALL_CLASSES:
        raise KeyError(f"Unknown case class {case_id}")
    rng = random.Random(_seed_int(case_id, seed))
    case = ALL_CLASSES[case_id](rng, num_agents)
    return TaskInstance(case, seed, num_agents)


def _seed_int(case_id: str, seed: int) -> int:
    """Stable per-class seed mixing: same numeric seed, different streams per
    class (avoids accidental correlation between classes)."""
    h = 0
    for ch in case_id:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return (h ^ (seed * 2654435761)) & 0xFFFFFFFF


def all_class_ids() -> list[str]:
    return sorted(ALL_CLASSES.keys())