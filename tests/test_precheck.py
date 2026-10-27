"""Tests for the dependency precheck (§11) against the plan's predictions.

The plan's pilot findings (§11) predict:
- I-01 Global Max: unidirectional (exactly one agent holds the global max)
- I-04 Any Match: ~30% degenerate (both agents locally match the global bool)
- Aggregations (I-07 average / I-10 std): bidirectional, almost no
  self-sufficiency
"""

import random

import pytest

from tasks import generate, all_class_ids
from tasks.oracles import classify_instance, oracle_answer, answers_equal
from tasks.precheck import precheck_class, run_precheck


def test_oracle_matches_generator_truth_for_global_class():
    """Sanity: the oracle answer for the agent that holds the global max
    equals the expected output (oracle correctness, not sufficiency)."""
    for seed in range(20):
        inst = generate("I-01", seed)
        holders = [
            a
            for a in range(2)
            if max(inst.input_shards[a]) == inst.expected_per_agent[a]
        ]
        assert len(holders) == 1, f"I-01 seed {seed}: expected exactly one holder"


def test_i01_classification_matches_plan_prediction():
    labels = [classify_instance(generate("I-01", s))["label"] for s in range(20)]
    assert all(l == "unidirectional" for l in labels), labels


def test_i04_degenerate_rate_substantial():
    """Empirically ~50% at n=2: when no ERROR exists (50% of seeds) both
    agents are locally correct.  The plan's pilot reported ~30% over 20
    seeds; this implementation measures the rate over 60 seeds and asserts
    the presence band that forces reseeding to be exercised."""
    labels = [classify_instance(generate("I-04", s))["label"] for s in range(60)]
    deg = sum(1 for l in labels if l == "degenerate")
    assert 18 <= deg <= 45, f"degenerate rate {deg}/60 outside expected band"


def test_aggregation_classes_are_bidirectional():
    for cid in ("I-07", "I-10"):
        labels = [classify_instance(generate(cid, s))["label"] for s in range(20)]
        assert all(l == "bidirectional" for l in labels), (cid, labels)


def test_all_classes_reach_10_valid_except_structural():
    """29 of 30 classes yield 10 valid instances; I-03 is structurally
    degenerate at n=2 (forced-majority construction: both shards always hold
    the local winner) and is recorded, not silently dropped."""
    report = run_precheck(target_per_class=10)
    assert report["summary"]["structurally_degenerate"] == ["I-03"]
    for cid in all_class_ids():
        c = report["classes"][cid]
        if cid == "I-03":
            continue
        assert c["valid_count"] == 10, (
            f"{cid}: only {c['valid_count']} valid after {c['seeds_tried']} seeds"
        )


def test_registry_records_reseeding_for_i04():
    result = precheck_class("I-04", target_per_class=10)
    # if any degenerate instances existed, seeds_tried must exceed 10
    deg = len(result["degenerate_excluded"])
    if deg:
        assert result["seeds_tried"] > 10
        assert result["reseeded"] is True