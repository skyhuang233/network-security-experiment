"""SILO-BENCH task classes ported to parameterized seeds (§14)."""

from .registry import TaskInstance, generate, all_class_ids, NUM_AGENTS

__all__ = ["TaskInstance", "generate", "all_class_ids", "NUM_AGENTS"]
