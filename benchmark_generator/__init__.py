"""SILO-BENCH benchmark generation stack.

Mirrors the upstream layout `benchmark_generator/`: the three `paradigm_*`
modules define the 30 task classes (I-01..III-30), and `generator` assembles
one TaskCase per (class, seed) into per-agent input shards + prompt templates,
carries the local oracles, and runs the §11 dependency precheck that keeps
only instances which actually require communication.

Consumed by `tasks.exp_a` / `tasks.exp_d` (via `generate(case_id, seed)` and
`load_registry`) and by `nse precheck` (`run_precheck` / `save_registry`);
exp_b / exp_c never touch it (known payloads, no task instances).
"""

from .paradigm_i import PARADIGM_I
from .paradigm_ii import PARADIGM_II
from .paradigm_iii import PARADIGM_III

__all__ = ["PARADIGM_I", "PARADIGM_II", "PARADIGM_III"]