"""Experiment scripts (A/B/C/D), nanochat-style standalone runs.

Task/benchmark generation lives in `benchmark_generator` (paradigm classes +
`generator` instance assembly & precheck). Each script keeps its own
module-level argparse + main flow; the LLM round loop (CaseRun/build_run,
prompts, tool-call protocol) is shared via `tasks/round_loop.py`, which is
safe to import because no script's main flow executes at import time.
"""

from benchmark_generator.generator import (
    TaskInstance,
    generate,
    all_class_ids,
    NUM_AGENTS,
)

__all__ = ["TaskInstance", "generate", "all_class_ids", "NUM_AGENTS"]