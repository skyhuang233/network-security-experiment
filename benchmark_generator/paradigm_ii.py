"""Paradigm II task generators (cases II-11, II-12), seed-parameterized.

Faithful port of benchmark_generator/paradigm_ii.py with `random.Random(seed)`
instances.  Case II-11 is SEGMENTED (per-agent outputs).
"""

from __future__ import annotations

import random
from typing import Any

from .paradigm_i import CaseGen, _shard


def ii11_prefix_sum(rng: random.Random, n: int) -> dict[str, Any]:
    data = [rng.randint(1, 50) for _ in range(n * 15)]
    cumsum, s = [], 0
    for v in data:
        s += v
        cumsum.append(s)
    description = """**Task: Prefix Sum (Cumulative Sum)**

Compute the prefix sum (cumulative sum) of the entire array.

**Agent Ordering:**
Agent 0 → Agent 1 → ... → Agent {max_id} (sequential segments)

**Your Data (Position {agent_id}):**
{input_shard}

**Algorithm:**
- prefix_sum[i] = sum of all elements from index 0 to i
- You need the TOTAL SUM from all previous agents to adjust your local prefix sums

**Output:**
Each agent submits THEIR portion of the prefix sum array.

**Communication Protocol:**
Topology: Chain (0 → 1 → 2 → ... → {max_id}). Communicate with neighbors only.
Each agent submits their own segment using submit_result(your_prefix_sum_segment)."""
    return dict(
        case_id="II-11", case_name="Prefix Sum", paradigm="Paradigm II",
        description=description, input_shards=_shard(data, n),
        expected_per_agent=_shard(cumsum, n), is_segmented=True,
    )


def ii12_moving_average(rng: random.Random, n: int) -> dict[str, Any]:
    data = [rng.randint(1, 100) for _ in range(n * 15)]
    expected = [
        round((data[i - 1] + data[i] + data[i + 1]) / 3.0, 2)
        for i in range(1, len(data) - 1)
    ]
    description = """**Task: Moving Average (Window=3)**

Compute a moving average with window size 3 for all valid positions.

**Agent Ordering:**
Agents in a LINE: 0 ↔ 1 ↔ 2 ↔ ... ↔ {max_id} (consecutive data)

**Your Data (Position {agent_id}):**
{input_shard}

**Algorithm:**
For each element (except first and last globally): avg = (prev + curr + next) / 3
- You need BOUNDARY elements from neighbors to compute averages at segment edges

**Output:**
The complete moving average array (excluding first and last positions).

**Communication Protocol:**
Topology: Chain (0 ↔ 1 ↔ ... ↔ {max_id}). Exchange boundaries with neighbors.
All agents must submit the same final answer using submit_result(global_moving_averages)."""
    return dict(
        case_id="II-12", case_name="Moving Average", paradigm="Paradigm II",
        description=description, input_shards=_shard(data, n),
        expected_per_agent=[expected] * n, is_segmented=False,
    )


PARADIGM_II: dict[str, CaseGen] = {
    "II-11": ii11_prefix_sum,
    "II-12": ii12_moving_average,
}
