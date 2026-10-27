"""Metrics computed from a real (scripted) run's persisted measurement JSON."""

import json

import pytest

from network_security_experiment.evaluation.metrics import compute_metrics
from network_security_experiment.harness.job import CaseRunConfig, build_run
from network_security_experiment.llm import ScriptedProvider
from tasks import generate

from .test_engine import _scripted_provider  # noqa: E402


@pytest.mark.parametrize("method", ["M1", "M2", "M3"])
def test_metrics_on_persisted_run(method, tmp_path):
    inst = generate("I-01", 0)
    config = CaseRunConfig(
        case_run_id=f"metrics-{method}", case_id="I-01", seed=0,
        method=method, max_rounds=10, key_set_size=128,
    )
    run = build_run(config, inst.user_prompts, _scripted_provider(), tmp_path)
    run.run()

    meas = json.loads((tmp_path / config.case_run_id / "measurement.json").read_text())
    m = compute_metrics(meas)

    # every chunk was delivered in this cooperative script
    assert m["effective_throughput"]["bidirectional_bits"] > 0
    for d, lat in m["communication_latency"].items():
        assert lat["censored_chunks"] == 0
        assert lat["latency_seconds_median"] is not None
    assert m["bit_error_rate"]["ber_total"] == 0.0
    assert m["resource_overhead"]["storage_op_layer"]["total_storage_ops"] > 0
    # round-tripped keys (str) are handled
    assert set(m["effective_throughput"]["per_direction"]) == {"0->1", "1->0"}