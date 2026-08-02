"""Coverage-focused tests for tempo_dag.verification.golden_trace."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tempo_dag.verification.golden_trace import (
    TRACE_SCHEMA_VERSION,
    GoldenTraceError,
    GoldenTraceRecorder,
    GoldenTraceValidator,
    diff_traces,
    load_golden_trace,
)
from tempo_dag.verification.temporal_parity import (
    TemporalExecutionTrace,
    TemporalTraceStep,
)


def _step(timestep, inputs=None, outputs=None, state=None):
    def _arrays(payload):
        return {
            name: np.asarray(value, dtype=np.float64)
            for name, value in (payload or {}).items()
        }

    return TemporalTraceStep(
        timestep=timestep,
        inputs=_arrays(inputs),
        outputs=_arrays(outputs),
        state=_arrays(state),
    )


def _trace(*steps):
    return TemporalExecutionTrace(tuple(steps))


def _payload(**overrides):
    payload = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "metadata": {},
        "steps": [{"timestep": 0, "inputs": {}, "outputs": {"y": [1.0]}, "state": {}}],
    }
    payload.update(overrides)
    return payload


def test_write_json_round_trips_through_loader(tmp_path) -> None:
    trace = _trace(
        _step(0, inputs={"x": [1.0]}, outputs={"y": [2.0]}, state={"s": [0.5]})
    )
    path = tmp_path / "golden.json"

    GoldenTraceRecorder().write_json(path, trace, metadata={"case": "round_trip"})
    loaded = load_golden_trace(path)

    assert loaded.metadata == {"case": "round_trip"}
    assert loaded.schema_version == TRACE_SCHEMA_VERSION
    assert loaded.steps[0].outputs["y"].tolist() == [2.0]
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_diff_traces_reports_length_mismatch_only() -> None:
    reference = _trace(_step(0), _step(1))
    candidate = _trace(_step(0))

    diffs = diff_traces(reference, candidate)

    assert len(diffs) == 1
    diff = diffs[0]
    assert diff.timestep == -1
    assert diff.section == "trace"
    assert diff.metric == "length"
    assert diff.expected == 2
    assert diff.actual == 1


def test_diff_traces_reports_timestep_mismatch() -> None:
    diffs = diff_traces(_trace(_step(0)), _trace(_step(5)))

    assert any(diff.name == "timestep" and diff.metric == "value" for diff in diffs)


def test_load_golden_trace_rejects_non_mapping_payload(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2]), encoding="utf-8")

    with pytest.raises(GoldenTraceError, match="must be a mapping"):
        load_golden_trace(path)


def test_validate_rejects_unsupported_schema_version() -> None:
    with pytest.raises(GoldenTraceError, match="unsupported schema_version"):
        GoldenTraceValidator().validate(_payload(schema_version="0.0"), _payload())


def test_validate_rejects_non_mapping_metadata() -> None:
    with pytest.raises(GoldenTraceError, match="metadata must be a mapping"):
        GoldenTraceValidator().validate(_payload(metadata=[1]), _payload())


def test_validate_rejects_non_list_steps() -> None:
    with pytest.raises(GoldenTraceError, match="steps must be a list"):
        GoldenTraceValidator().validate(_payload(steps="nope"), _payload())


def test_validate_rejects_non_mapping_step() -> None:
    with pytest.raises(GoldenTraceError, match="each step must be a mapping"):
        GoldenTraceValidator().validate(_payload(steps=[42]), _payload())


def test_validate_rejects_non_integer_timestep() -> None:
    with pytest.raises(GoldenTraceError, match="timestep must be an integer"):
        GoldenTraceValidator().validate(_payload(steps=[{"timestep": "0"}]), _payload())


def test_validate_rejects_non_mapping_section() -> None:
    bad_step = {"timestep": 0, "inputs": [1.0]}
    with pytest.raises(GoldenTraceError, match="inputs must be a mapping"):
        GoldenTraceValidator().validate(_payload(steps=[bad_step]), _payload())


def test_diff_named_arrays_reports_missing_and_unexpected_names() -> None:
    reference = _trace(_step(0, outputs={"a": [1.0]}))
    candidate = _trace(_step(0, outputs={"b": [1.0]}))

    diffs = diff_traces(reference, candidate)

    findings = {(diff.metric, diff.name) for diff in diffs}
    assert ("missing", "a") in findings
    assert ("unexpected", "b") in findings


def test_diff_named_arrays_reports_shape_mismatch_without_value_diff() -> None:
    reference = _trace(_step(0, outputs={"a": [1.0, 2.0]}))
    candidate = _trace(_step(0, outputs={"a": [1.0]}))

    diffs = diff_traces(reference, candidate)

    assert len(diffs) == 1
    diff = diffs[0]
    assert diff.metric == "shape"
    assert diff.expected == [2]
    assert diff.actual == [1]
    assert diff.max_abs_diff is None
    assert "max_abs_diff" not in diff.to_dict()


def test_diff_values_mismatch_reports_max_abs_diff() -> None:
    reference = _trace(_step(0, outputs={"a": [1.0, 2.0]}))
    candidate = _trace(_step(0, outputs={"a": [1.0, 2.5]}))

    diffs = diff_traces(reference, candidate, atol=0.1)

    assert len(diffs) == 1
    diff = diffs[0]
    assert diff.metric == "values"
    assert diff.max_abs_diff == pytest.approx(0.5)
    assert diff.to_dict()["max_abs_diff"] == pytest.approx(0.5)


def test_diff_values_within_atol_pass() -> None:
    reference = _trace(_step(0, outputs={"a": [1.0]}))
    candidate = _trace(_step(0, outputs={"a": [1.4]}))

    assert diff_traces(reference, candidate, atol=0.5) == []


def test_validator_accepts_golden_trace_and_execution_trace_inputs() -> None:
    trace = _trace(_step(0, inputs={"x": [1.0]}, outputs={"y": [2.0]}))
    golden = GoldenTraceRecorder().record(trace, metadata={"case": "mixed_inputs"})

    report = GoldenTraceValidator().validate(golden, trace)

    assert report["pass"] is True
    assert report["num_steps"] == 1
    assert report["diffs"] == []
    assert report["summary"] == "Golden trace matches candidate trace."
