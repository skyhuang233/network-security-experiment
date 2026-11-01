"""Task instances, local oracles, and the dependency precheck (§11, §14).

`generate(case_id, seed)` builds one two-agent instance. An oracle says what
each agent could answer from its own shard alone, and the precheck keeps only
instances that actually require communication.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

from .paradigm_i import PARADIGM_I
from .paradigm_ii import PARADIGM_II
from .paradigm_iii import PARADIGM_III

ALL_CLASSES: dict[str, Any] = {**PARADIGM_I, **PARADIGM_II, **PARADIGM_III}

NUM_AGENTS = 2  # fixed by §14: every class at n=2


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


# The recorded benchmark.  Six classes, one from each communication pattern
# (low-payload aggregation, set exchange, one-way dependency, boundary
# exchange, data redistribution, graph traversal), five instances each.
# 6 classes x 5 seeds x 3 carriers = 90 task runs, doubled by the
# separated-backend negative control to the 180 recorded runs.
BENCHMARK_CLASSES: list[str] = ["I-06", "I-08", "II-11", "II-12", "III-21", "III-24"]
BENCHMARK_SEEDS_PER_CLASS = 5


Oracle = Callable[[TaskInstance, int], Any]


def _close(a, b, tol: float = 0.01) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def _norm(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_norm(x) for x in v)
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


# ---------------------------------------------------------------- Paradigm I


def i06(inst: TaskInstance, agent: int) -> Any:
    x = 0
    for v in inst.input_shards[agent]:
        x ^= v
    return x


def i08(inst: TaskInstance, agent: int) -> Any:
    return len(set(inst.input_shards[agent]))


# --------------------------------------------------------------- Paradigm II


def ii11(inst: TaskInstance, agent: int) -> Any:
    # local prefix sums WITHOUT the global offset: cannot match the expected
    # segment unless agent==0; oracle computes the local-only attempt
    s, out = 0, []
    for v in inst.input_shards[agent]:
        s += v
        out.append(s)
    return out


def ii12(inst: TaskInstance, agent: int) -> Any:
    # local moving averages (window 3) without boundary elements
    d = inst.input_shards[agent]
    out = []
    for i in range(1, len(d) - 1):
        out.append(round((d[i - 1] + d[i] + d[i + 1]) / 3.0, 2))
    return out


# ------------------------------------------------------------- Paradigm III


def iii21(inst: TaskInstance, agent: int) -> Any:
    # local sort of own shard; compare against expected segment
    return sorted(inst.input_shards[agent])


def iii24(inst: TaskInstance, agent: int) -> Any:
    # BFS from node 0 over local edges only: distances dict over seen nodes
    edges = inst.input_shards[agent]
    adj: dict[int, list[int]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    dist = {0: 0}
    queue = [0]
    while queue:
        node = queue.pop(0)
        for nb in adj.get(node, []):
            if nb not in dist:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    return dist


ORACLES: dict[str, Oracle] = {
    "I-06": i06, "I-08": i08,
    "II-11": ii11, "II-12": ii12,
    "III-21": iii21, "III-24": iii24,
}


def oracle_answer(inst: TaskInstance, agent: int) -> Any:
    return ORACLES[inst.case_id](inst, agent)


def answers_equal(got: Any, expected: Any) -> bool:
    """Compare an oracle answer to the expected per-agent value (tolerant)."""
    if got is None or expected is None:
        return got is None and expected is None
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return _close(got, expected, tol=0.01)
    return _norm(got) == _norm(expected)


def classify_instance(inst: TaskInstance) -> dict[str, Any]:
    """Per-instance dependency layer label (§11)."""
    sufficiency = []
    for agent in range(inst.num_agents):
        got = oracle_answer(inst, agent)
        sufficiency.append(answers_equal(got, inst.expected_per_agent[agent]))
    if all(sufficiency):
        label, direction = "degenerate", "none"
    elif any(sufficiency):
        label = "unidirectional"
        src = sufficiency.index(True)
        dst = sufficiency.index(False)
        direction = f"{src}->{dst}"
    else:
        label, direction = "bidirectional", "A<->B"
    return {
        "instance_id": inst.instance_id,
        "case_id": inst.case_id,
        "seed": inst.seed,
        "sufficient_agents": sufficiency,
        "label": label,
        "direction": direction,
    }


def precheck_class(
    case_id: str,
    target_per_class: int = 10,
    seed_start: int = 0,
    max_seeds: int = 1000,
) -> dict[str, Any]:
    """Sweep seeds for one class until `target_per_class` valid instances."""
    valid: list[dict[str, Any]] = []
    degenerate: list[dict[str, Any]] = []
    seeds_tried = 0
    seed = seed_start
    while len(valid) < target_per_class and seed < seed_start + max_seeds:
        inst = generate(case_id, seed)
        result = classify_instance(inst)
        result["seed_tried_index"] = seeds_tried
        seeds_tried += 1
        if result["label"] == "degenerate":
            degenerate.append(result)
        else:
            valid.append(result)
        seed += 1
    return {
        "case_id": case_id,
        "valid_count": len(valid),
        "seeds_tried": seeds_tried,
        "reseeded": seeds_tried > target_per_class,
        "structurally_degenerate": len(valid) < target_per_class,
        "valid": valid,
        "degenerate_excluded": degenerate,
        "label_counts": {
            "unidirectional": sum(1 for v in valid if v["label"] == "unidirectional"),
            "bidirectional": sum(1 for v in valid if v["label"] == "bidirectional"),
        },
    }


def run_precheck(
    target_per_class: int = 10,
    class_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Full precheck over all (or selected) classes.

    Classes that stay degenerate across the entire seed sweep (at n=2 a
    construction can make both shards locally sufficient, so both agents
    are always self-sufficient) are recorded as
    `structurally_degenerate` and excluded from the main experiment rather
    than infinitely reseeded (§11: the reseeding process is recorded).
    """
    report: dict[str, Any] = {"target_per_class": target_per_class, "classes": {}}
    for cid in class_ids or all_class_ids():
        report["classes"][cid] = precheck_class(cid, target_per_class)
    report["summary"] = {
        "classes_complete": sum(
            1
            for c in report["classes"].values()
            if c["valid_count"] == target_per_class
        ),
        "structurally_degenerate": [
            cid
            for cid, c in report["classes"].items()
            if c["structurally_degenerate"]
        ],
        "total_valid": sum(c["valid_count"] for c in report["classes"].values()),
        "total_degenerate_excluded": sum(
            len(c["degenerate_excluded"]) for c in report["classes"].values()
        ),
    }
    return report


def save_registry(report: dict[str, Any], path: str | Path) -> None:
    """Persist the seed -> instance registry (§14 seed→instance mapping)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))


def load_registry(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def valid_seeds_for(report: dict[str, Any], case_id: str) -> list[int]:
    return [entry["seed"] for entry in report["classes"][case_id]["valid"]]
