"""Paradigm II task generators (cases II-11..II-20), seed-parameterized.

Faithful port of benchmark_generator/paradigm_ii.py with `random.Random(seed)`
instances.  Cases II-11, II-14, II-20 are SEGMENTED (per-agent outputs).
"""

from __future__ import annotations

import random
from typing import Any

from .paradigm_i import CaseGen, _shard, _shard_str


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


def ii13_longest_palindrome(rng: random.Random, n: int) -> dict[str, Any]:
    chars = "abcdefghijklmnopqrstuvwxyz"
    data = [rng.choice(chars) for _ in range(n * 15)]
    palindrome = "racecar"
    pos = rng.randint(0, len(data) - len(palindrome))
    for j, c in enumerate(palindrome):
        data[pos + j] = c
    string = "".join(data)

    def lp_len(s: str) -> int:
        if not s:
            return 0
        best = 1
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                sub = s[i:j]
                if sub == sub[::-1] and len(sub) > best:
                    best = len(sub)
        return best

    expected = lp_len(string)
    description = """**Task: Longest Palindrome (Cross-Boundary)**

Find the length of the longest palindromic substring, including palindromes spanning agent boundaries.

**Agent Ordering:**
Agents hold CONSECUTIVE parts of one string: 0 → 1 → ... → {max_id}

**Your Data (Position {agent_id}):**
String segment: {input_shard}

**Algorithm:**
1. Find longest palindrome within your local segment
2. Exchange BOUNDARY CHARACTERS with neighbors to detect cross-boundary palindromes
3. Coordinate to find the global maximum length

**Output:**
A single integer representing the length of the longest palindrome.

**Communication Protocol:**
Topology: Chain (0 ↔ 1 ↔ ... ↔ {max_id}). Exchange prefix/suffix with neighbors.
All agents must submit the same final answer using submit_result(max_palindrome_length)."""
    return dict(
        case_id="II-13", case_name="Longest Palindrome", paradigm="Paradigm II",
        description=description, input_shards=_shard_str(string, n),
        expected_per_agent=[expected] * n, is_segmented=False,
    )


def ii14_life_game(rng: random.Random, n: int) -> dict[str, Any]:
    steps = 5
    state = [rng.randint(0, 1) for _ in range(n * 10)]

    def step_once(arr):
        new = arr[:]
        for i in range(len(arr)):
            left = arr[i - 1] if i - 1 >= 0 else 0
            right = arr[i + 1] if i + 1 < len(arr) else 0
            s = left + arr[i] + right
            new[i] = 1 if s == 1 else 0
        return new

    cur = state
    for _ in range(steps):
        cur = step_once(cur)
    description = f"""**Task: 1D Life Game (Cellular Automaton)**

Simulate a 1D cellular automaton for {steps} steps.

**Agent Ordering:**
Agents arranged sequentially: 0 ↔ 1 ↔ 2 ↔ ... ↔ {{max_id}}

**Your Data (Position {{agent_id}}):**
Initial segment: {{input_shard}}

**Algorithm:**
- Each cell's next state depends on (left, self, right)
- Rule: cell becomes 1 iff exactly one of (left, self, right) is 1
- Exchange boundary cells with neighbors at EACH step

**Output:**
Each agent submits their own final segment after {steps} steps.

**Communication Protocol:**
Topology: Chain (0 ↔ 1 ↔ ... ↔ {{max_id}}). Repeated boundary exchange.
Each agent submits their own segment using submit_result(your_final_segment)."""
    return dict(
        case_id="II-14", case_name="1D Life Game", paradigm="Paradigm II",
        description=description, input_shards=_shard(state, n),
        expected_per_agent=_shard(cur, n), is_segmented=True,
    )


def ii15_pattern_search(rng: random.Random, n: int) -> dict[str, Any]:
    chars = ["A", "B", "C", "D", "X", "Y", "Z"]
    data = [rng.choice(chars) for _ in range(n * 20)]
    for _ in range(2):
        pos = rng.randint(0, len(data) - 15)
        data[pos] = "A"
        data[pos + rng.randint(1, 5)] = "B"
        data[pos + rng.randint(6, 10)] = "C"
    string = "".join(data)

    def count_subseq(s: str) -> int:
        count, st = 0, 0
        for ch in s:
            if st == 0 and ch == "A":
                st = 1
            elif st == 1 and ch == "B":
                st = 2
            elif st == 2 and ch == "C":
                count += 1
                st = 0
        return count

    expected = count_subseq(string)
    description = """**Task: Subsequence Pattern Search**

Count how many times the subsequence pattern "A...B...C" appears in the global string.
(A followed eventually by B followed eventually by C)

**Agent Ordering:**
Agents hold CONSECUTIVE parts: 0 → 1 → 2 → ... → {max_id}

**Your Data (Position {agent_id}):**
String segment: {input_shard}

**Algorithm:**
Track state machine: 0 (looking for A) → 1 (found A) → 2 (found B) → complete on C
- State must be PASSED FORWARD through the chain (pipeline pattern)

**Output:**
A single integer representing the count of "A...B...C" patterns.

**Communication Protocol:**
Topology: Chain (0 → 1 → 2 → ... → {max_id}). Pass state forward.
All agents must submit the same final answer using submit_result(total_pattern_count)."""
    return dict(
        case_id="II-15", case_name="Pattern Search", paradigm="Paradigm II",
        description=description, input_shards=_shard_str(string, n),
        expected_per_agent=[expected] * n, is_segmented=False,
    )


def ii16_trapping_rain(rng: random.Random, n: int) -> dict[str, Any]:
    data = [rng.randint(0, 10) for _ in range(n * 10)]
    for i in range(1, len(data) - 1, 5):
        data[i] = rng.randint(0, 3)

    def trap(h):
        if not h:
            return 0
        m = len(h)
        lm = [0] * m
        rm = [0] * m
        lm[0] = h[0]
        for i in range(1, m):
            lm[i] = max(lm[i - 1], h[i])
        rm[m - 1] = h[m - 1]
        for i in range(m - 2, -1, -1):
            rm[i] = max(rm[i + 1], h[i])
        return sum(min(lm[i], rm[i]) - h[i] for i in range(m))

    expected = trap(data)
    description = """**Task: Trapping Rain Water**

Given an elevation map, compute how much water can be trapped after raining.

**Agent Ordering:**
Agents hold CONSECUTIVE segments: 0 ↔ 1 ↔ ... ↔ {max_id}

**Your Data (Position {agent_id}):**
Elevation heights: {input_shard}

**Algorithm:**
- water_at_i = min(max_height_to_left, max_height_to_right) - height[i]
- Requires TWO PASSES: left-to-right (get left_max), right-to-left (get right_max)

**Output:**
A single integer representing the total trapped water units.

**Communication Protocol:**
Topology: Chain (0 ↔ 1 ↔ ... ↔ {max_id}). Bidirectional pass.
All agents must submit the same final answer using submit_result(total_water)."""
    return dict(
        case_id="II-16", case_name="Trapping Rain", paradigm="Paradigm II",
        description=description, input_shards=_shard(data, n),
        expected_per_agent=[expected] * n, is_segmented=False,
    )


def ii17_diff_array(rng: random.Random, n: int) -> dict[str, Any]:
    data = [rng.randint(1, 100) for _ in range(n * 15)]
    expected = [data[i] - data[i - 1] for i in range(1, len(data))]
    description = """**Task: Difference Array**

Compute the difference array: diff[i] = arr[i] - arr[i-1] for all i >= 1.

**Agent Ordering:**
Agents hold CONSECUTIVE elements: 0 → 1 → ... → {max_id}

**Your Data (Position {agent_id}):**
{input_shard}

**Algorithm:**
- Compute differences within your segment
- First element of your segment needs the LAST element from previous agent

**Output:**
The complete difference array.

**Communication Protocol:**
Topology: Chain (0 → 1 → ... → {max_id}). Get boundary from left neighbor.
All agents must submit the same final answer using submit_result(global_diff_array)."""
    return dict(
        case_id="II-17", case_name="Diff Array", paradigm="Paradigm II",
        description=description, input_shards=_shard(data, n),
        expected_per_agent=[expected] * n, is_segmented=False,
    )


def ii18_list_ranking(rng: random.Random, n: int) -> dict[str, Any]:
    total = n * 8
    nodes = list(range(total))
    rng.shuffle(nodes)  # first shuffle: logical chain order
    next_pointers = {nodes[i]: nodes[i + 1] for i in range(len(nodes) - 1)}
    next_pointers[nodes[-1]] = -1
    true_head = nodes[0]
    rng.shuffle(nodes)  # second shuffle: distribution to agents
    shard_nodes = _shard(nodes, n)
    shards = [
        [{"id": nid, "next": next_pointers[nid]} for nid in part]
        for part in shard_nodes
    ]
    distances = {}
    current, ordered = true_head, []
    while current != -1:
        distances[current] = len(ordered)
        ordered.append(current)
        current = next_pointers[current]
    description = """**Task: List Ranking (Compute Rank from Head)**

Reconstruct a distributed linked list and compute each node's rank from the head.

**Data Distribution:**
- Total nodes: {total_size}, SCATTERED across agents
- Each node has "id" and "next" pointer (-1 for end)
- Forms ONE logical chain

**Your Data:**
Agent {{agent_id}} holds: {{input_shard}}

**Algorithm:**
1. Reconstruct the logical chain starting from head node
2. Compute each node's rank (head=0, next=1, ...)

**Output:**
Dictionary mapping node_id to its rank: {{node_id: rank, ...}}

**Communication Protocol:**
Topology: Share pointers to reconstruct chain, then broadcast ranks.
All agents must submit the same final answer using submit_result({{node_id: rank, ...}})."""
    d = description.format(total_size=total)
    return dict(
        case_id="II-18", case_name="List Ranking", paradigm="Paradigm II",
        description=d, input_shards=shards,
        expected_per_agent=[distances] * n, is_segmented=False,
    )


def ii19_merge_neighbors(rng: random.Random, n: int) -> dict[str, Any]:
    data = sorted(rng.randint(1, 50) for _ in range(n * 15))
    expected = [v for i, v in enumerate(data) if i == 0 or data[i] != data[i - 1]]
    description = """**Task: Deduplicate Consecutive Elements**

Given sorted consecutive segments, remove consecutive duplicate elements globally.

**Agent Ordering:**
Agents hold SORTED CONSECUTIVE segments: 0 → 1 → ... → {max_id}

**Your Data (Position {agent_id}):**
Sorted segment: {input_shard}

**Algorithm:**
1. Remove consecutive duplicates within your segment
2. Check boundaries with neighbors to remove cross-boundary duplicates

**Output:**
A sorted array with no consecutive duplicates.

**Communication Protocol:**
Topology: Chain (0 ↔ 1 ↔ ... ↔ {max_id}). Compare boundaries with neighbors.
All agents must submit the same final answer using submit_result(global_deduped_array)."""
    return dict(
        case_id="II-19", case_name="Merge Neighbors", paradigm="Paradigm II",
        description=description, input_shards=_shard(data, n),
        expected_per_agent=[expected] * n, is_segmented=False,
    )


def ii20_pipeline_hash(rng: random.Random, n: int) -> dict[str, Any]:
    data = [f"Block_{i}_Data" for i in range(n * 5)]

    def simple_hash(prev, block):
        return (sum(ord(c) for c in block) % 10000 ^ prev) % 10000

    expected, cur = [], 0
    for block in data:
        cur = simple_hash(cur, block)
        expected.append(cur)
    description = """**Task: Pipeline Hash (Blockchain Simulation)**

Compute a hash chain where each block's hash depends on the previous block's hash.

**Agent Ordering:**
Agents hold CONSECUTIVE blocks: 0 → 1 → 2 → ... → {max_id}

**Your Data (Position {agent_id}):**
Blocks: {input_shard}

**Algorithm:**
hash[i] = (sum(ASCII_values(block[i])) XOR hash[i-1]) % 10000
hash[0] = 0 (genesis)
- Agent i needs the FINAL HASH from Agent i-1 to start computing

**Output:**
Each agent submits their portion of hash values.

**Communication Protocol:**
Topology: Chain (0 → 1 → 2 → ... → {max_id}). Strict sequential dependency.
Each agent submits their own segment using submit_result(your_computed_hashes)."""
    return dict(
        case_id="II-20", case_name="Pipeline Hash", paradigm="Paradigm II",
        description=description, input_shards=_shard(data, n),
        expected_per_agent=_shard(expected, n), is_segmented=True,
    )


PARADIGM_II: dict[str, CaseGen] = {
    "II-11": ii11_prefix_sum,
    "II-12": ii12_moving_average,
    "II-13": ii13_longest_palindrome,
    "II-14": ii14_life_game,
    "II-15": ii15_pattern_search,
    "II-16": ii16_trapping_rain,
    "II-17": ii17_diff_array,
    "II-18": ii18_list_ranking,
    "II-19": ii19_merge_neighbors,
    "II-20": ii20_pipeline_hash,
}