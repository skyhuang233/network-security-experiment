"""Information-dependency precheck pipeline (§11).

Avoids mistaking "accidentally self-sufficient" instances for communication.
For each of the 30 classes, sweep seeds, run the local oracle per agent,
classify (degenerate / unidirectional / bidirectional), and reseed until the
class has `target_per_class` VALID (non-degenerate) instances — the reseeding
process is recorded (§11: "每类 10 实例可能因剔除双自足而不足，需按类补种").

Deterministic, offline, run once before formal experiments.  Output: a JSON
registry consumed by the experiment runners.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .oracles import classify_instance
from .registry import NUM_AGENTS, TaskInstance, all_class_ids, generate


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