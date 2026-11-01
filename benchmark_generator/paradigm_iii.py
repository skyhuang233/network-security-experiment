"""Paradigm III task generators (cases III-21, III-24), seed-parameterized.

Faithful port of benchmark_generator/paradigm_iii.py with `random.Random(seed)`
instances.  Case III-21 is SEGMENTED (per-agent outputs).
"""

from __future__ import annotations

import random
from typing import Any

from .paradigm_i import CaseGen, _shard


def iii21_distributed_sort(rng: random.Random, n: int) -> dict[str, Any]:
    data = [rng.randint(1, 1000) for _ in range(n * 20)]
    rng.shuffle(data)
    sorted_data = sorted(data)
    description = """**Task: Distributed Sorting**

Sort the global dataset so that Agent 0 holds smallest values, Agent {max_id} holds largest.

**Data Distribution:**
RANDOM - your data may belong to other agents' final ranges.

**Your Data (RANDOM):**
Agent {agent_id} holds: {input_shard}

**Algorithm:**
1. Determine value ranges for each agent
2. SHUFFLE data between agents
3. Each agent sorts their final range

**Output:**
Each agent submits their sorted portion (concatenated result should be fully sorted).

**Communication Protocol:**
Topology: All-to-All / Hash-based shuffle. Global data exchange required.
Each agent submits their own segment using submit_result(your_sorted_range)."""
    return dict(
        case_id="III-21", case_name="Distributed Sort", paradigm="Paradigm III",
        description=description, input_shards=_shard(data, n),
        expected_per_agent=_shard(sorted_data, n), is_segmented=True,
    )


def iii24_bfs_distance(rng: random.Random, n: int) -> dict[str, Any]:
    num_nodes = n * 8
    graph = {i: [] for i in range(num_nodes)}
    edges = []
    for _ in range(num_nodes * 2):
        u = rng.randint(0, num_nodes - 1)
        v = rng.randint(0, num_nodes - 1)
        if u != v and v not in graph[u]:
            graph[u].append(v)
            graph[v].append(u)
            edges.append((u, v))
    rng.shuffle(edges)
    distances = {i: -1 for i in range(num_nodes)}
    distances[0] = 0
    queue = [0]
    while queue:
        node = queue.pop(0)
        for nb in graph[node]:
            if distances[nb] == -1:
                distances[nb] = distances[node] + 1
                queue.append(nb)
    description = f"""**Task: BFS Distance (Single-Source Shortest Path)**

Compute shortest distance from source node 0 to all nodes in an unweighted graph.

**Data Distribution:**
Total nodes: {num_nodes}. Edges RANDOMLY distributed. Source: node 0.

**Your Data (RANDOM edges):**
Agent {{{{agent_id}}}} holds: {{{{input_shard}}}}

**Algorithm:**
BFS wave propagation. Messages must propagate along edges held by different agents.

**Output:**
Dictionary mapping node_id to distance_from_source (-1 if unreachable).

**Communication Protocol:**
Topology: Multi-hop message passing along graph structure.
All agents must submit the same final answer using submit_result(complete_distance_dict)."""
    return dict(
        case_id="III-24", case_name="BFS Distance", paradigm="Paradigm III",
        description=description, input_shards=_shard(edges, n),
        expected_per_agent=[distances] * n, is_segmented=False,
    )


PARADIGM_III: dict[str, CaseGen] = {
    "III-21": iii21_distributed_sort,
    "III-24": iii24_bfs_distance,
}
