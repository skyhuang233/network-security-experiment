"""Paradigm I task generators (cases I-06, I-08), seed-parameterized.

Faithful port of references/silo-bench/acl26-silo-bench/benchmark_generator/
paradigm_i.py with one change required by design doc §14: the module-level
`random.seed(42)` is replaced by a `random.Random(seed)` instance so each
(class, seed) pair yields an independent instance.  Data distributions,
sizes and computation order follow the original generator.

Each generator returns a TaskCase dict:
    case_id, case_name, paradigm, description (template with {agent_id} /
    {input_shard}), input_shards (list per agent), expected_per_agent
    (list per agent), is_segmented.
"""

from __future__ import annotations

import random
from typing import Any, Callable

CaseGen = Callable[[random.Random, int], dict[str, Any]]


def _shard(data: list, num_agents: int) -> list[list]:
    chunk = len(data) // num_agents
    out = []
    for i in range(num_agents):
        start = i * chunk
        end = start + chunk if i < num_agents - 1 else len(data)
        out.append(data[start:end])
    return out


def _shard_str(s: str, num_agents: int) -> list[str]:
    parts = _shard(list(s), num_agents)
    return ["".join(p) for p in parts]


def i06_checksum(rng: random.Random, n: int) -> dict[str, Any]:
    data = [rng.randint(0, 255) for _ in range(n * 20)]
    expected = 0
    for val in data:
        expected ^= val
    description = """**Task: XOR Checksum**

Compute the XOR checksum of all data bytes across all agents.

**Your Data:**
You (Agent {agent_id}) hold: {input_shard}

**Algorithm:**
1. Compute local XOR: local_xor = byte1 ^ byte2 ^ ... ^ byteN
2. Coordinate with other agents to compute global XOR
3. Property: XOR is associative and commutative, so global_xor = xor1 ^ xor2 ^ ... ^ xorN

**Output:**
A single integer (0-255) representing the global XOR checksum.

**Communication Protocol:**
Topology: Star/Tree aggregation → broadcast.
All agents must submit the same final answer using submit_result(global_checksum)."""
    return dict(
        case_id="I-06", case_name="Checksum", paradigm="Paradigm I",
        description=description, input_shards=_shard(data, n),
        expected_per_agent=[expected] * n, is_segmented=False,
    )


def i08_set_union_size(rng: random.Random, n: int) -> dict[str, Any]:
    total = n * 30
    unique_elements = list(range(total // 2))
    data = [rng.choice(unique_elements) for _ in range(total)]
    description = """**Task: Global Distinct Count**

Count the total number of DISTINCT (unique) elements across all agents' data.

**Your Data:**
You (Agent {agent_id}) hold (may contain duplicates): {input_shard}

**Algorithm:**
1. Identify unique elements in your local data: local_set = set(your_data)
2. Coordinate with other agents to find the global set of distinct elements
3. Note: Elements may appear in multiple agents' data - need global deduplication

**Output:**
A single integer representing the total count of distinct elements.

**Communication Protocol:**
Topology: Star/Tree aggregation or hash-based partitioning → broadcast.
All agents must submit the same final answer using submit_result(global_distinct_count)."""
    return dict(
        case_id="I-08", case_name="Set Union Size", paradigm="Paradigm I",
        description=description, input_shards=_shard(data, n),
        expected_per_agent=[len(set(data))] * n, is_segmented=False,
    )


PARADIGM_I: dict[str, CaseGen] = {
    "I-06": i06_checksum,
    "I-08": i08_set_union_size,
}
