"""Paradigm III task generators (cases III-21..III-30), seed-parameterized.

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


def iii22_median(rng: random.Random, n: int) -> dict[str, Any]:
    data = [rng.randint(1, 1000) for _ in range(n * 25)]
    rng.shuffle(data)
    s = sorted(data)
    median_idx = len(s) // 2
    expected = s[median_idx]
    description = f"""**Task: Global Median (Iterative)**

Find the median value without collecting all data (use iterative consensus).

**Data Distribution:**
RANDOM across agents. Total elements: {len(data)}, median at position {median_idx}.

**Your Data (RANDOM):**
Agent {{{{agent_id}}}} holds: {{{{input_shard}}}}

**Algorithm (Iterative):**
1. Each agent computes local median/quantiles
2. Estimate global pivot from local medians
3. Each agent counts: values < pivot, = pivot, > pivot
4. Determine which range contains median, repeat until convergence

**Output:**
A single integer representing the global median.

**Communication Protocol:**
Topology: All-to-All. Multiple rounds of coordination required.
All agents must submit the same final answer using submit_result(median_value)."""
    return dict(
        case_id="III-22", case_name="Median of Medians", paradigm="Paradigm III",
        description=description, input_shards=_shard(data, n),
        expected_per_agent=[expected] * n, is_segmented=False,
    )


def iii23_graph_components(rng: random.Random, n: int) -> dict[str, Any]:
    num_nodes = n * 10
    edges = []
    for _ in range(num_nodes * 2):
        u = rng.randint(0, num_nodes - 1)
        v = rng.randint(0, num_nodes - 1)
        if u != v:
            edges.append((u, v))
    rng.shuffle(edges)
    parent = list(range(num_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv
    expected = len({find(i) for i in range(num_nodes)})
    description = f"""**Task: Graph Connected Components**

Find the number of connected components in an undirected graph with randomly distributed edges.

**Data Distribution:**
Total nodes: {num_nodes}. Edges RANDOMLY distributed across agents.

**Your Data (RANDOM edges):**
Agent {{{{agent_id}}}} holds: {{{{input_shard}}}} (each edge is tuple (u, v))

**Algorithm:**
Use Union-Find. Coordinate to merge components connected through edges held by different agents.

**Output:**
A single integer representing the number of connected components.

**Communication Protocol:**
Topology: All-to-All or Hash-based. Share component membership information.
All agents must submit the same final answer using submit_result(component_count)."""
    return dict(
        case_id="III-23", case_name="Graph Components", paradigm="Paradigm III",
        description=description, input_shards=_shard(edges, n),
        expected_per_agent=[expected] * n, is_segmented=False,
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


def iii25_kmeans(rng: random.Random, n: int) -> dict[str, Any]:
    K = 3
    total = n * 20
    points = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(total)]
    centroids = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(K)]
    rng.shuffle(points)
    clusters = [[] for _ in range(K)]
    for p in points:
        k = min(range(K), key=lambda j: (p[0] - centroids[j][0]) ** 2 + (p[1] - centroids[j][1]) ** 2)
        clusters[k].append(p)
    new_centroids = []
    for k in range(K):
        if clusters[k]:
            cx = sum(p[0] for p in clusters[k]) / len(clusters[k])
            cy = sum(p[1] for p in clusters[k]) / len(clusters[k])
            new_centroids.append((round(cx, 2), round(cy, 2)))
        else:
            new_centroids.append((round(centroids[k][0], 2), round(centroids[k][1], 2)))
    description = f"""**Task: K-Means Clustering (One Iteration)**

Perform ONE iteration of K-Means to update {K} cluster centroids.

**Data Distribution:**
2D points RANDOMLY distributed. Initial centroids: {centroids}

**Your Data (RANDOM points):**
Agent {{{{agent_id}}}} holds: {{{{input_shard}}}}

**Algorithm:**
1. Assign each point to nearest centroid
2. Compute new centroids as mean of assigned points
3. Aggregate statistics across agents (sum_x, sum_y, count per cluster)

**Output:**
List of {K} centroids: [(x0, y0), (x1, y1), (x2, y2)]

**Communication Protocol:**
Topology: All-to-All or cluster-based shuffle. Aggregate by cluster ID.
All agents must submit the same final answer using submit_result([(x0, y0), ...])."""
    return dict(
        case_id="III-25", case_name="K-Means Iteration", paradigm="Paradigm III",
        description=description, input_shards=_shard(points, n),
        expected_per_agent=[new_centroids] * n, is_segmented=False,
    )


def iii26_global_distinct(rng: random.Random, n: int) -> dict[str, Any]:
    total = n * 30
    unique_elements = list(range(total // 2))
    data = [rng.choice(unique_elements) for _ in range(total)]
    rng.shuffle(data)
    description = """**Task: Global Distinct Count (Hash-Based)**

Count the total DISTINCT elements using hash-based deduplication.

**Data Distribution:**
RANDOM with duplicates. Same element may appear in multiple agents.

**Your Data (RANDOM with duplicates):**
Agent {agent_id} holds: {input_shard}

**Algorithm:**
1. Hash-based partitioning: bucket_id = hash(element) % num_agents
2. SHUFFLE elements to owner agents
3. Each agent deduplicates their buckets
4. Aggregate counts

**Output:**
A single integer representing the global distinct count.

**Communication Protocol:**
Topology: All-to-All shuffle by hash.
All agents must submit the same final answer using submit_result(global_distinct_count)."""
    return dict(
        case_id="III-26", case_name="Global Distinct", paradigm="Paradigm III",
        description=description, input_shards=_shard(data, n),
        expected_per_agent=[len(set(data))] * n, is_segmented=False,
    )


def iii27_collaborative_filtering(rng: random.Random, n: int) -> dict[str, Any]:
    num_users = n * 3
    num_items = n * 3
    K = 3
    user_vectors = {i: [rng.uniform(0, 1) for _ in range(5)] for i in range(num_users)}
    item_vectors = {i: [rng.uniform(0, 1) for _ in range(5)] for i in range(num_items)}
    all_data = [("user", k, v) for k, v in user_vectors.items()] + [
        ("item", k, v) for k, v in item_vectors.items()
    ]
    rng.shuffle(all_data)
    chunk = len(all_data) // n
    shards = []
    for i in range(n):
        start = i * chunk
        end = start + chunk if i < n - 1 else len(all_data)
        agent_data = {"users": {}, "items": {}}
        for dtype, did, vec in all_data[start:end]:
            agent_data["users" if dtype == "user" else "items"][did] = vec
        shards.append(agent_data)
    scores = []
    for uid, uvec in user_vectors.items():
        for iid, ivec in item_vectors.items():
            s = sum(a * b for a, b in zip(uvec, ivec))
            scores.append((uid, iid, round(s, 4)))
    scores.sort(key=lambda x: x[2], reverse=True)
    expected = scores[:K]
    description = f"""**Task: Collaborative Filtering (Top User-Item Pairs)**

Find the top {K} user-item pairs with highest compatibility scores (dot product).

**Data Distribution:**
User/Item vectors RANDOMLY distributed. Each has 5-dim feature vector.

**Your Data (RANDOM):**
Agent {{{{agent_id}}}} holds: {{{{input_shard}}}} (dict with "users" and "items" keys)

**Algorithm:**
1. Compute dot products for ALL (user, item) pairs
2. Find top {K} pairs with highest scores
3. score(user, item) = dot_product(user_vector, item_vector)

**Output:**
List of top {K} tuples: [(user_id, item_id, score), ...]

**Communication Protocol:**
Topology: All-to-All. This is O(users × items) computation.
All agents must submit the same final answer using submit_result([(uid, iid, score), ...])."""
    return dict(
        case_id="III-27", case_name="Collaborative Filtering", paradigm="Paradigm III",
        description=description, input_shards=shards,
        expected_per_agent=[expected] * n, is_segmented=False,
    )


def iii28_pagerank(rng: random.Random, n: int) -> dict[str, Any]:
    num_nodes = n * 8
    graph = {i: [] for i in range(num_nodes)}
    edges = []
    for _ in range(num_nodes * 2):
        u = rng.randint(0, num_nodes - 1)
        v = rng.randint(0, num_nodes - 1)
        if u != v and v not in graph[u]:
            graph[u].append(v)
            edges.append((u, v))
    rng.shuffle(edges)
    damping = 0.85
    initial = 1.0 / num_nodes
    new_ranks = {i: (1 - damping) / num_nodes for i in range(num_nodes)}
    for node in range(num_nodes):
        out = len(graph[node])
        if out > 0:
            contrib = damping * initial / out
            for nb in graph[node]:
                new_ranks[nb] += contrib
    expected = {k: round(v, 6) for k, v in new_ranks.items()}
    description = f"""**Task: PageRank (One Iteration)**

Compute one iteration of PageRank on a directed graph.

**Data Distribution:**
Total nodes: {num_nodes}. Edges RANDOMLY distributed. Initial ranks: uniform.

**Your Data (RANDOM edges):**
Agent {{{{agent_id}}}} holds: {{{{input_shard}}}}

**Algorithm:**
rank'[v] = (1-d)/{num_nodes} + d × Σ(rank[u]/out_degree[u]) for all u→v
where d = 0.85 (damping factor)

**Output:**
Dictionary mapping node_id to rank_value.

**Communication Protocol:**
Topology: All-to-All with message routing along edges.
All agents must submit the same final answer using submit_result(complete_rank_dict)."""
    return dict(
        case_id="III-28", case_name="PageRank Step", paradigm="Paradigm III",
        description=description, input_shards=_shard(edges, n),
        expected_per_agent=[expected] * n, is_segmented=False,
    )


def iii29_load_balance(rng: random.Random, n: int) -> dict[str, Any]:
    total_tasks = n * 20
    tasks = []
    for i in range(total_tasks):
        if i % 10 == 0:
            duration = rng.randint(50, 100)
        else:
            duration = rng.randint(1, 10)
        tasks.append({"id": i, "duration": duration})
    heavy = [t for t in tasks if t["id"] % 10 == 0]
    light = [t for t in tasks if t["id"] % 10 != 0]
    shards = [heavy]
    light_chunk = len(light) // (n - 1) if n > 1 else 0
    for i in range(1, n):
        start = (i - 1) * light_chunk
        end = start + light_chunk if i < n - 1 else len(light)
        shards.append(light[start:end])
    total_duration = sum(t["duration"] for t in tasks)
    target = total_duration / n
    sorted_tasks = sorted(tasks, key=lambda t: t["duration"], reverse=True)
    loads = [0] * n
    for t in sorted_tasks:
        m = min(range(n), key=lambda i: loads[i])
        loads[m] += t["duration"]
    expected = {
        "variance": round(sum((l - target) ** 2 for l in loads) / n, 2),
        "max_load": max(loads),
        "min_load": min(loads),
    }
    description = """**Task: Load Balancing (Deterministic Greedy LPT)**

Redistribute tasks using a deterministic algorithm (LPT - Longest Processing Time first).

**Data Distribution:**
Severely UNBALANCED. Total tasks: {total_tasks}, total duration: {total_duration}.

**Your Data (UNBALANCED):**
Agent {{agent_id}} holds: {{input_shard}}

**Algorithm (MUST follow exactly):**
1. Collect ALL tasks globally
2. Sort by duration descending
3. Greedily assign to agent with minimum load (tie-break: smallest agent id)
4. Compute: variance, max_load, min_load

**Output:**
Statistics dict: {{"variance": X, "max_load": Y, "min_load": Z}}

**Communication Protocol:**
Topology: All-to-All share tasks, then compute and broadcast.
All agents must submit the same final answer using submit_result(stats_dict)."""
    d = description.format(total_tasks=total_tasks, total_duration=total_duration)
    return dict(
        case_id="III-29", case_name="Load Balance", paradigm="Paradigm III",
        description=d, input_shards=shards,
        expected_per_agent=[expected] * n, is_segmented=False,
    )


def iii30_matrix_multiply(rng: random.Random, n: int) -> dict[str, Any]:
    dim = n
    A = [[rng.randint(1, 10) for _ in range(dim)] for _ in range(dim)]
    B = [[rng.randint(1, 10) for _ in range(dim)] for _ in range(dim)]
    rows_per = dim // n
    shards = []
    for i in range(n):
        start = i * rows_per
        end = start + rows_per if i < n - 1 else dim
        shards.append(
            {
                "type": "rows_and_full_B",
                "A_rows": A[start:end],
                "A_row_indices": list(range(start, end)),
                "B_matrix": B,
            }
        )
    C = [
        [sum(A[i][k] * B[k][j] for k in range(dim)) for j in range(dim)]
        for i in range(dim)
    ]
    description = """**Task: Distributed Matrix Multiplication**

Compute C = A × B where matrix A is row-partitioned across agents.

**Data Distribution:**
Matrix {dim}×{dim}. Each agent has rows of A and complete B.

**Your Data:**
Agent {{agent_id}} holds: {{input_shard}}
- A_rows: Your assigned rows from A
- A_row_indices: Which rows you have
- B_matrix: Complete matrix B

**Algorithm:**
C[i,j] = Σ(A[i,k] × B[k,j]) for k = 0 to {dim_minus_1}
You can compute your assigned rows locally, then share.

**Output:**
The complete {dim}×{dim} result matrix C.

**Communication Protocol:**
Topology: All-to-All for sharing computed rows.
All agents must submit the same final answer using submit_result(C)."""
    d = description.format(dim=dim, dim_minus_1=dim - 1)
    return dict(
        case_id="III-30", case_name="Matrix Multiply", paradigm="Paradigm III",
        description=d, input_shards=shards,
        expected_per_agent=[C] * n, is_segmented=False,
    )


PARADIGM_III: dict[str, CaseGen] = {
    "III-21": iii21_distributed_sort,
    "III-22": iii22_median,
    "III-23": iii23_graph_components,
    "III-24": iii24_bfs_distance,
    "III-25": iii25_kmeans,
    "III-26": iii26_global_distinct,
    "III-27": iii27_collaborative_filtering,
    "III-28": iii28_pagerank,
    "III-29": iii29_load_balance,
    "III-30": iii30_matrix_multiply,
}