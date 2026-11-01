"""Task instances, local oracles, and the dependency precheck (§11, §14).

`generate(case_id, seed)` builds one two-agent instance. An oracle says what
each agent could answer from its own shard alone, and the precheck keeps only
instances that actually require communication.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any, Callable

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


def i01(inst: TaskInstance, agent: int) -> Any:
    return max(inst.input_shards[agent])


def i02(inst: TaskInstance, agent: int) -> Any:
    return sum(1 for w in inst.input_shards[agent] if w == "apple")


def i03(inst: TaskInstance, agent: int) -> Any:
    votes = inst.input_shards[agent]
    counts: dict[str, int] = {}
    for c in votes:
        counts[c] = counts.get(c, 0) + 1
    return max(counts, key=counts.get) if counts else None


def i04(inst: TaskInstance, agent: int) -> Any:
    return any("ERROR" in s for s in inst.input_shards[agent])


def i05(inst: TaskInstance, agent: int) -> Any:
    return sum(1 for x in inst.input_shards[agent] if 100 <= x <= 500)


def i06(inst: TaskInstance, agent: int) -> Any:
    x = 0
    for v in inst.input_shards[agent]:
        x ^= v
    return x


def i07(inst: TaskInstance, agent: int) -> Any:
    shard = inst.input_shards[agent]
    return round(sum(shard) / len(shard), 2)


def i08(inst: TaskInstance, agent: int) -> Any:
    return len(set(inst.input_shards[agent]))


def i09(inst: TaskInstance, agent: int) -> Any:
    return sorted(inst.input_shards[agent], reverse=True)[:10]


def i10(inst: TaskInstance, agent: int) -> Any:
    shard = inst.input_shards[agent]
    return round(statistics.pstdev(shard), 2)


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


def ii13(inst: TaskInstance, agent: int) -> Any:
    seg = "".join(inst.input_shards[agent]) if isinstance(inst.input_shards[agent][0], str) else "".join(map(str, inst.input_shards[agent]))
    best = 1 if seg else 0
    for i in range(len(seg)):
        for j in range(i + 1, len(seg) + 1):
            sub = seg[i:j]
            if sub == sub[::-1] and len(sub) > best:
                best = len(sub)
    return best


def ii14(inst: TaskInstance, agent: int) -> Any:
    # simulate 5 steps on the local segment with zero boundaries (no neighbors)
    arr = list(inst.input_shards[agent])
    for _ in range(5):
        new = arr[:]
        for i in range(len(arr)):
            left = arr[i - 1] if i - 1 >= 0 else 0
            right = arr[i + 1] if i + 1 < len(arr) else 0
            s = left + arr[i] + right
            new[i] = 1 if s == 1 else 0
        arr = new
    return arr


def ii15(inst: TaskInstance, agent: int) -> Any:
    seg = "".join(inst.input_shards[agent])
    count, st = 0, 0
    for ch in seg:
        if st == 0 and ch == "A":
            st = 1
        elif st == 1 and ch == "B":
            st = 2
        elif st == 2 and ch == "C":
            count += 1
            st = 0
    return count


def ii16(inst: TaskInstance, agent: int) -> Any:
    h = list(inst.input_shards[agent])
    m = len(h)
    if m == 0:
        return 0
    lm = [0] * m
    rm = [0] * m
    lm[0] = h[0]
    for i in range(1, m):
        lm[i] = max(lm[i - 1], h[i])
    rm[m - 1] = h[m - 1]
    for i in range(m - 2, -1, -1):
        rm[i] = max(rm[i + 1], h[i])
    return sum(min(lm[i], rm[i]) - h[i] for i in range(m))


def ii17(inst: TaskInstance, agent: int) -> Any:
    d = inst.input_shards[agent]
    out = []
    for i in range(1, len(d)):
        out.append(d[i] - d[i - 1])
    if agent > 0:
        out = [None] + out  # first diff needs the neighbor's last element
    return out if agent == 0 else None


def ii18(inst: TaskInstance, agent: int) -> Any:
    # chain reconstruction from a local shard alone is impossible unless the
    # shard happens to contain the full chain
    nodes = inst.input_shards[agent]
    nxt = {n["id"]: n["next"] for n in nodes}
    ids = {n["id"] for n in nodes}
    heads = [i for i in ids if not any(n["next"] == i for n in nodes)]
    if len(heads) != 1:
        return None
    order, cur = [], heads[0]
    while cur != -1 and cur in nxt:
        order.append(cur)
        cur = nxt[cur]
    if cur != -1 or len(order) != len(nxt):
        return None
    return {nid: rank for rank, nid in enumerate(order)}


def ii19(inst: TaskInstance, agent: int) -> Any:
    d = inst.input_shards[agent]
    out = [v for i, v in enumerate(d) if i == 0 or d[i] != d[i - 1]]
    if agent > 0:
        out = [None] + out  # cross-boundary dedup needs neighbor's last value
    return out if agent == 0 else None


def ii20(inst: TaskInstance, agent: int) -> Any:
    # hash chain without the incoming hash: computable only for agent 0
    if agent > 0:
        return None
    out, cur = [], 0
    for block in inst.input_shards[agent]:
        cur = (sum(ord(c) for c in block) % 10000 ^ cur) % 10000
        out.append(cur)
    return out


# ------------------------------------------------------------- Paradigm III


def iii21(inst: TaskInstance, agent: int) -> Any:
    # local sort of own shard; compare against expected segment
    return sorted(inst.input_shards[agent])


def iii22(inst: TaskInstance, agent: int) -> Any:
    shard = sorted(inst.input_shards[agent])
    return shard[len(shard) // 2]


def iii23(inst: TaskInstance, agent: int) -> Any:
    edges = inst.input_shards[agent]
    nodes = {u for e in edges for u in e}
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv
    return len({find(n) for n in nodes}) if nodes else 0


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


def iii25(inst: TaskInstance, agent: int) -> Any:
    # one K-means iteration using only local points with GLOBAL initial
    # centroids (initial centroids are public in the task description)
    import re

    m = re.search(r"Initial centroids: \[\((.*)\)\]", inst.user_prompts[0])
    if not m:
        raise ValueError("cannot recover initial centroids from prompt")
    cts = []
    for pair in m.group(1).split("), ("):
        x, y = pair.split(", ")
        cts.append((float(x), float(y)))
    K = len(cts)
    clusters = [[] for _ in range(K)]
    for p in inst.input_shards[agent]:
        k = min(range(K), key=lambda j: (p[0] - cts[j][0]) ** 2 + (p[1] - cts[j][1]) ** 2)
        clusters[k].append(p)
    out = []
    for k in range(K):
        if clusters[k]:
            out.append(
                (
                    round(sum(p[0] for p in clusters[k]) / len(clusters[k]), 2),
                    round(sum(p[1] for p in clusters[k]) / len(clusters[k]), 2),
                )
            )
        else:
            out.append((round(cts[k][0], 2), round(cts[k][1], 2)))
    return out


def iii26(inst: TaskInstance, agent: int) -> Any:
    return len(set(inst.input_shards[agent]))


def iii27(inst: TaskInstance, agent: int) -> Any:
    data = inst.input_shards[agent]
    users, items = data["users"], data["items"]
    scores = []
    for uid, uvec in users.items():
        for iid, ivec in items.items():
            scores.append((uid, iid, round(sum(a * b for a, b in zip(uvec, ivec)), 4)))
    if not scores:
        return None
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:3]


def iii28(inst: TaskInstance, agent: int) -> Any:
    # one PageRank iteration over local edges only
    edges = inst.input_shards[agent]
    nodes = {u for e in edges for u in e}
    num_nodes = inst.num_agents * 8  # node count is public in the task
    out_deg: dict[int, int] = {}
    for u, v in edges:
        out_deg[u] = out_deg.get(u, 0) + 1
    ranks = {i: (1 - 0.85) / num_nodes for i in range(num_nodes)}
    initial = 1.0 / num_nodes
    for u, v in edges:
        ranks[v] += 0.85 * initial / out_deg[u]
    return {k: round(v, 6) for k, v in ranks.items()}


def iii29(inst: TaskInstance, agent: int) -> Any:
    tasks = inst.input_shards[agent]
    n = inst.num_agents
    all_tasks = [t for t in tasks]
    total_duration = sum(t["duration"] for t in all_tasks)
    target = total_duration / n
    sorted_tasks = sorted(all_tasks, key=lambda t: t["duration"], reverse=True)
    loads = [0] * n
    for t in sorted_tasks:
        m = min(range(n), key=lambda i: loads[i])
        loads[m] += t["duration"]
    return {
        "variance": round(sum((l - target) ** 2 for l in loads) / n, 2),
        "max_load": max(loads),
        "min_load": min(loads),
    }


def iii30(inst: TaskInstance, agent: int) -> Any:
    data = inst.input_shards[agent]
    A_rows, B = data["A_rows"], data["B_matrix"]
    dim = len(B)
    return [
        [sum(r[k] * B[k][j] for k in range(dim)) for j in range(dim)] for r in A_rows
    ]


ORACLES: dict[str, Oracle] = {
    "I-01": i01, "I-02": i02, "I-03": i03, "I-04": i04, "I-05": i05,
    "I-06": i06, "I-07": i07, "I-08": i08, "I-09": i09, "I-10": i10,
    "II-11": ii11, "II-12": ii12, "II-13": ii13, "II-14": ii14, "II-15": ii15,
    "II-16": ii16, "II-17": ii17, "II-18": ii18, "II-19": ii19, "II-20": ii20,
    "III-21": iii21, "III-22": iii22, "III-23": iii23, "III-24": iii24,
    "III-25": iii25, "III-26": iii26, "III-27": iii27, "III-28": iii28,
    "III-29": iii29, "III-30": iii30,
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

    Classes that stay degenerate across the entire seed sweep (e.g. I-03 at
    n=2: the forced-majority construction makes BOTH shards hold the local
    winner, so both agents are always self-sufficient) are recorded as
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
