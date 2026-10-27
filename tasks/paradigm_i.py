"""Paradigm I task generators (cases I-01..I-10), seed-parameterized.

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


def i01_global_max(rng: random.Random, n: int) -> dict[str, Any]:
    data = [rng.randint(-1000, 1000) for _ in range(n * 20)]
    description = """**Task: Global Maximum**

Find the GLOBAL MAXIMUM value across all agents' data.

**Your Data:**
You (Agent {agent_id}) hold: {input_shard}

**Algorithm:**
1. Compute the maximum in your local shard
2. Coordinate with other agents to determine the global maximum

**Output:**
A single integer representing the global maximum value.

**Communication Protocol:**
Topology: Star/Tree aggregation → broadcast.
All agents must submit the same final answer using submit_result(global_max)."""
    return dict(
        case_id="I-01", case_name="Global Max", paradigm="Paradigm I",
        description=description, input_shards=_shard(data, n),
        expected_per_agent=[max(data)] * n, is_segmented=False,
    )


def i02_word_frequency(rng: random.Random, n: int) -> dict[str, Any]:
    words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
    target = "apple"
    data = [rng.choice(words) for _ in range(n * 30)]
    for i in range(0, len(data), 10):
        data[i] = target
    description = f"""**Task: Word Frequency Count**

Count the TOTAL occurrences of the word "{target}" across all agents' data.

**Your Data:**
You (Agent {{agent_id}}) hold: {{input_shard}}

**Algorithm:**
1. Count how many times "{target}" appears in your local data
2. Coordinate with other agents to compute the global count

**Output:**
A single integer representing the total count of "{target}".

**Communication Protocol:**
Topology: Star/Tree aggregation → broadcast.
All agents must submit the same final answer using submit_result(total_count)."""
    return dict(
        case_id="I-02", case_name="Word Frequency", paradigm="Paradigm I",
        description=description, input_shards=_shard(data, n),
        expected_per_agent=[data.count(target)] * n, is_segmented=False,
    )


def i03_distributed_vote(rng: random.Random, n: int) -> dict[str, Any]:
    candidates = ["Candidate_A", "Candidate_B", "Candidate_C", "Candidate_D"]
    votes = [rng.choice(candidates) for _ in range(n * 25)]
    winner = rng.choice(candidates)
    for i in range(0, len(votes), 2):
        votes[i] = winner
    counts = {c: votes.count(c) for c in candidates}
    expected = max(counts, key=counts.get)
    description = """**Task: Distributed Voting**

Determine which candidate received the MOST votes across all agents' data.

**Your Data:**
You (Agent {agent_id}) hold: {input_shard}

**Algorithm:**
1. Count the votes for each candidate in your local shard
2. Collaborate with other agents to determine the global winner

**Output:**
A string with the candidate name who received the most votes (e.g., "Candidate_A").

**Communication Protocol:**
Topology: Star/Tree aggregation → broadcast.
All agents must submit the same final answer using submit_result("Candidate_X")."""
    return dict(
        case_id="I-03", case_name="Distributed Vote", paradigm="Paradigm I",
        description=description, input_shards=_shard(votes, n),
        expected_per_agent=[expected] * n, is_segmented=False,
    )


def i04_any_match(rng: random.Random, n: int) -> dict[str, Any]:
    strings = [
        "INFO: System started", "DEBUG: Loading config", "WARNING: Low memory",
        "SUCCESS: Task completed", "TRACE: Function called",
    ]
    data = [rng.choice(strings) for _ in range(n * 15)]
    has_error = rng.choice([True, False])
    if has_error:
        data[rng.randint(0, len(data) - 1)] = "ERROR: Critical failure detected"
    expected = any("ERROR" in s for s in data)
    description = """**Task: Error Detection**

Determine if ANY string in the global dataset contains the substring "ERROR".

**Your Data:**
You (Agent {agent_id}) hold: {input_shard}

**Algorithm:**
1. Check if any string in your local data contains "ERROR"
2. Coordinate with other agents to determine if ERROR exists anywhere
3. Early termination: Once ANY agent finds "ERROR", the answer is True

**Output:**
A boolean (True/False) indicating if "ERROR" was found anywhere.

**Communication Protocol:**
Topology: Star/Tree with early termination → broadcast.
All agents must submit the same final answer using submit_result(True/False)."""
    return dict(
        case_id="I-04", case_name="Any Match", paradigm="Paradigm I",
        description=description, input_shards=_shard(data, n),
        expected_per_agent=[expected] * n, is_segmented=False,
    )


def i05_range_count(rng: random.Random, n: int) -> dict[str, Any]:
    L, R = 100, 500
    data = [rng.randint(0, 1000) for _ in range(n * 30)]
    expected = sum(1 for x in data if L <= x <= R)
    description = f"""**Task: Range Count**

Count how many numbers in the global dataset fall within the range [{L}, {R}] (inclusive).

**Your Data:**
You (Agent {{agent_id}}) hold: {{input_shard}}

**Algorithm:**
1. Count how many numbers in your local data satisfy: {L} ≤ x ≤ {R}
2. Coordinate with other agents to compute the global count

**Output:**
A single integer representing the total count of numbers in range [{L}, {R}].

**Communication Protocol:**
Topology: Star/Tree aggregation → broadcast.
All agents must submit the same final answer using submit_result(total_count)."""
    return dict(
        case_id="I-05", case_name="Range Count", paradigm="Paradigm I",
        description=description, input_shards=_shard(data, n),
        expected_per_agent=[expected] * n, is_segmented=False,
    )


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


def i07_average_value(rng: random.Random, n: int) -> dict[str, Any]:
    data = [rng.uniform(0, 100) for _ in range(n * 25)]
    expected = round(sum(data) / len(data), 2)
    description = """**Task: Global Average**

Compute the average (mean) value of all numbers across all agents' data.

**Your Data:**
You (Agent {agent_id}) hold: {input_shard}

**Algorithm:**
1. Compute local_sum and local_count
2. Coordinate with other agents to compute global_sum and global_count
3. global_average = global_sum / global_count

**Output:**
A floating-point number representing the global average (precision: 2 decimal places).

**Communication Protocol:**
Topology: Star/Tree aggregation → broadcast.
All agents must submit the same final answer using submit_result(global_average)."""
    return dict(
        case_id="I-07", case_name="Average Value", paradigm="Paradigm I",
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


def i09_top_k_select(rng: random.Random, n: int) -> dict[str, Any]:
    K = 10
    data = [rng.randint(1, 1000) for _ in range(n * 30)]
    expected = sorted(data, reverse=True)[:K]
    description = f"""**Task: Top-{K} Selection**

Find the {K} largest values across all agents' data.

**Your Data:**
You (Agent {{agent_id}}) hold: {{input_shard}}

**Algorithm:**
1. Find the top {K} values in your local data
2. Coordinate with other agents to find the global top-{K}
3. Each agent only needs to send at most {K} values to a coordinator

**Output:**
A list of {K} integers, sorted in descending order.

**Communication Protocol:**
Topology: Star/Tree aggregation → broadcast.
All agents must submit the same final answer using submit_result([val1, val2, ..., val{K}])."""
    return dict(
        case_id="I-09", case_name="Top-K Select", paradigm="Paradigm I",
        description=description, input_shards=_shard(data, n),
        expected_per_agent=[expected] * n, is_segmented=False,
    )


def i10_std_deviation(rng: random.Random, n: int) -> dict[str, Any]:
    data = [rng.uniform(0, 100) for _ in range(n * 30)]
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    expected = round(variance ** 0.5, 2)
    description = """**Task: Standard Deviation (Two-Phase)**

Compute the population standard deviation of all numbers across all agents' data.

**Your Data:**
You (Agent {agent_id}) hold: {input_shard}

**Algorithm (Two Phases):**
**Phase 1: Compute Global Mean**
- Calculate local_sum and local_count
- Coordinate to compute global_mean = total_sum / total_count

**Phase 2: Compute Variance**
- Calculate local variance contribution: sum((x - global_mean)² for x in local_data)
- Coordinate to sum all variance contributions
- Compute std_dev = sqrt(total_variance / total_count)

**Output:**
A floating-point number (precision: 2 decimal places).

**Communication Protocol:**
Topology: Star/Tree with two-phase protocol → broadcast after each phase.
All agents must submit the same final answer using submit_result(std_dev)."""
    return dict(
        case_id="I-10", case_name="Standard Deviation", paradigm="Paradigm I",
        description=description, input_shards=_shard(data, n),
        expected_per_agent=[expected] * n, is_segmented=False,
    )


PARADIGM_I: dict[str, CaseGen] = {
    "I-01": i01_global_max,
    "I-02": i02_word_frequency,
    "I-03": i03_distributed_vote,
    "I-04": i04_any_match,
    "I-05": i05_range_count,
    "I-06": i06_checksum,
    "I-07": i07_average_value,
    "I-08": i08_set_union_size,
    "I-09": i09_top_k_select,
    "I-10": i10_std_deviation,
}