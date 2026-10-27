"""End-to-end harness integration tests (round loop + channels + measurement).

Uses the dev-only ScriptedProvider (real LLMs are used in recorded runs);
these tests verify the deterministic plumbing: round visibility, delivery
recording, submissions, termination, and measurement artifacts.
"""

import json

import pytest

from network_security_experiment.harness.job import CaseRunConfig, build_run
from network_security_experiment.llm import ScriptedProvider
from tasks import generate


def _scripted_provider():
    # Both agents read (receive) every round — required by fire-and-forget
    # delivery on M3, where a skipped read round permanently loses that
    # segment.  A publishes its local max, B answers with its own, both
    # submit the global max (999).
    return ScriptedProvider(
        scripts={
            0: [
                (
                    "[[\n  <tool>send_payload</tool>\n  <parameters>\n"
                    "    <data>my local max is 827</data>\n  </parameters>\n]]\n"
                    "[[\n  <tool>wait</tool>\n  <parameters></parameters>\n]]"
                ),
                "[[\n  <tool>receive</tool>\n  <parameters></parameters>\n]]\n"
                "[[\n  <tool>wait</tool>\n  <parameters></parameters>\n]]",
                "[[\n  <tool>receive</tool>\n  <parameters></parameters>\n]]\n"
                "[[\n  <tool>wait</tool>\n  <parameters></parameters>\n]]",
                "[[\n  <tool>receive</tool>\n  <parameters></parameters>\n]]\n"
                "[[\n  <tool>wait</tool>\n  <parameters></parameters>\n]]",
                "[[\n  <tool>receive</tool>\n  <parameters></parameters>\n]]\n"
                "[[\n  <tool>submit_result</tool>\n  <parameters>\n"
                "    <answer>999</answer>\n  </parameters>\n]]",
            ],
            1: [
                "[[\n  <tool>receive</tool>\n  <parameters></parameters>\n]]\n"
                "[[\n  <tool>wait</tool>\n  <parameters></parameters>\n]]",
                (
                    "[[\n  <tool>receive</tool>\n  <parameters></parameters>\n]]\n"
                    "[[\n  <tool>send_payload</tool>\n  <parameters>\n"
                    "    <data>ack: my local max is 999</data>\n  </parameters>\n]]\n"
                    "[[\n  <tool>wait</tool>\n  <parameters></parameters>\n]]"
                ),
                "[[\n  <tool>receive</tool>\n  <parameters></parameters>\n]]\n"
                "[[\n  <tool>submit_result</tool>\n  <parameters>\n"
                "    <answer>999</answer>\n  </parameters>\n]]",
            ],
        }
    )


@pytest.mark.parametrize("method", ["M1", "M2", "M3"])
def test_round_loop_end_to_end(method, tmp_path):
    inst = generate("I-01", 0)
    config = CaseRunConfig(
        case_run_id=f"smoke-{method}",
        case_id="I-01",
        seed=0,
        method=method,
        max_rounds=10,
        key_set_size=128,
    )
    provider = _scripted_provider()
    run = build_run(config, inst.user_prompts, provider, tmp_path)
    result = run.run()

    # both agents submitted
    assert result["submissions"] == {0: 999, 1: 999}
    # rounds recorded
    assert 2 <= result["rounds"] <= 10
    # measurement artifacts persisted
    meas = json.loads((tmp_path / config.case_run_id / "measurement.json").read_text())
    assert meas["method"] == method
    # at least one chunk was sent by each agent
    senders = {c["sender"] for c in meas["chunks"].values()}
    assert senders == {0, 1}
    # deliveries recorded: A->B and B->A both have delivered chunks
    for c in meas["chunks"].values():
        assert c["delivered"], f"chunk {c} never delivered"

    # cross-check throughput data exists
    assert meas["t_start"] is not None and meas["t_end"] is not None
    assert meas["overhead"]["tool_calls"] >= 6  # send/receive calls


def test_max_rounds_termination(tmp_path):
    # agent loops wait forever -> terminates at max_rounds
    provider = ScriptedProvider(
        scripts={
            0: ["[[\n  <tool>wait</tool>\n  <parameters></parameters>\n]]"],
            1: ["[[\n  <tool>wait</tool>\n  <parameters></parameters>\n]]"],
        }
    )
    inst = generate("I-01", 0)
    config = CaseRunConfig(
        case_run_id="smoke-maxrounds",
        case_id="I-01",
        seed=0,
        method="M1",
        max_rounds=3,
    )
    run = build_run(config, inst.user_prompts, provider, tmp_path)
    result = run.run()
    assert result["rounds"] == 3
    assert result["submissions"] == {0: None, 1: None}