from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .temporal_parity import TemporalExecutionTrace, TemporalTraceStep

TRACE_SCHEMA_VERSION = "1.0"


class GoldenTraceError(Exception):
    """Raised when a golden trace payload is malformed or incompatible."""


@dataclass(frozen=True)
class TraceDiff:
    """Single mismatch found while comparing two temporal traces."""

    timestep: int
    section: str
    name: str
    metric: str
    expected: object
    actual: object
    max_abs_diff: float | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "timestep": self.timestep,
            "section": self.section,
            "name": self.name,
            "metric": self.metric,
            "expected": self.expected,
            "actual": self.actual,
        }
        if self.max_abs_diff is not None:
            payload["max_abs_diff"] = self.max_abs_diff
        return payload


@dataclass(frozen=True)
class GoldenTrace:
    """JSON-serializable temporal verification artifact."""

    schema_version: str
    metadata: dict[str, object]
    steps: tuple[TemporalTraceStep, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "metadata": dict(self.metadata),
            "steps": [step.to_dict() for step in self.steps],
        }

    def to_execution_trace(self) -> TemporalExecutionTrace:
        return TemporalExecutionTrace(self.steps)


class GoldenTraceRecorder:
    """Captures a temporal execution trace into the golden trace payload."""

    def record(
        self,
        trace: TemporalExecutionTrace,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> GoldenTrace:
        return GoldenTrace(
            schema_version=TRACE_SCHEMA_VERSION,
            metadata=dict(metadata or {}),
            steps=trace.steps,
        )

    def write_json(
        self,
        path: str | Path,
        trace: TemporalExecutionTrace,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        golden_trace = self.record(trace, metadata=metadata)
        output_path = Path(path)
        output_path.write_text(
            json.dumps(golden_trace.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class GoldenTraceValidator:
    """Validates runtime traces against stored golden traces."""

    def validate(
        self,
        reference: GoldenTrace | Mapping[str, object] | TemporalExecutionTrace,
        candidate: GoldenTrace | Mapping[str, object] | TemporalExecutionTrace,
        *,
        atol: float = 0.0,
    ) -> dict[str, object]:
        reference_trace = _coerce_to_trace(reference)
        candidate_trace = _coerce_to_trace(candidate)
        diffs = diff_traces(reference_trace, candidate_trace, atol=atol)
        return {
            "pass": not diffs,
            "schema_version": TRACE_SCHEMA_VERSION,
            "num_steps": len(reference_trace.steps),
            "diffs": [diff.to_dict() for diff in diffs],
            "summary": (
                "Golden trace matches candidate trace."
                if not diffs
                else f"Golden trace mismatch count: {len(diffs)}"
            ),
        }


def diff_traces(
    reference: TemporalExecutionTrace,
    candidate: TemporalExecutionTrace,
    *,
    atol: float = 0.0,
) -> list[TraceDiff]:
    diffs: list[TraceDiff] = []
    if len(reference.steps) != len(candidate.steps):
        diffs.append(
            TraceDiff(
                timestep=-1,
                section="trace",
                name="steps",
                metric="length",
                expected=len(reference.steps),
                actual=len(candidate.steps),
            )
        )
        return diffs

    for reference_step, candidate_step in zip(
        reference.steps, candidate.steps, strict=True
    ):
        if reference_step.timestep != candidate_step.timestep:
            diffs.append(
                TraceDiff(
                    timestep=reference_step.timestep,
                    section="trace",
                    name="timestep",
                    metric="value",
                    expected=reference_step.timestep,
                    actual=candidate_step.timestep,
                )
            )
        diffs.extend(
            _diff_named_arrays(
                reference_step.timestep,
                "inputs",
                reference_step.inputs,
                candidate_step.inputs,
                atol=atol,
            )
        )
        diffs.extend(
            _diff_named_arrays(
                reference_step.timestep,
                "outputs",
                reference_step.outputs,
                candidate_step.outputs,
                atol=atol,
            )
        )
        diffs.extend(
            _diff_named_arrays(
                reference_step.timestep,
                "state",
                reference_step.state,
                candidate_step.state,
                atol=atol,
            )
        )
    return diffs


def load_golden_trace(path: str | Path) -> GoldenTrace:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise GoldenTraceError("golden trace payload must be a mapping")
    return _coerce_to_golden_trace(payload)


def _coerce_to_trace(
    value: GoldenTrace | Mapping[str, object] | TemporalExecutionTrace,
) -> TemporalExecutionTrace:
    if isinstance(value, TemporalExecutionTrace):
        return value
    if isinstance(value, GoldenTrace):
        return value.to_execution_trace()
    return _coerce_to_golden_trace(value).to_execution_trace()


def _coerce_to_golden_trace(value: Mapping[str, object]) -> GoldenTrace:
    schema_version = value.get("schema_version")
    if schema_version != TRACE_SCHEMA_VERSION:
        raise GoldenTraceError(
            f"unsupported schema_version {schema_version!r}; "
            f"expected {TRACE_SCHEMA_VERSION!r}"
        )

    metadata_value = value.get("metadata", {})
    if not isinstance(metadata_value, Mapping):
        raise GoldenTraceError("metadata must be a mapping")

    steps_value = value.get("steps")
    if not isinstance(steps_value, list):
        raise GoldenTraceError("steps must be a list")

    steps = tuple(_parse_step(step) for step in steps_value)
    return GoldenTrace(
        schema_version=TRACE_SCHEMA_VERSION,
        metadata=dict(metadata_value),
        steps=steps,
    )


def _parse_step(value: object) -> TemporalTraceStep:
    if not isinstance(value, Mapping):
        raise GoldenTraceError("each step must be a mapping")

    timestep = value.get("timestep")
    if not isinstance(timestep, int):
        raise GoldenTraceError("step timestep must be an integer")

    return TemporalTraceStep(
        timestep=timestep,
        inputs=_parse_named_arrays(value.get("inputs", {}), section="inputs"),
        outputs=_parse_named_arrays(value.get("outputs", {}), section="outputs"),
        state=_parse_named_arrays(value.get("state", {}), section="state"),
    )


def _parse_named_arrays(value: object, *, section: str) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise GoldenTraceError(f"{section} must be a mapping")
    return {str(name): _parse_numeric_array(item) for name, item in value.items()}


def _parse_numeric_array(value: object) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:  # pragma: no cover
        raise GoldenTraceError("trace values must be numeric arrays") from exc


def _diff_named_arrays(
    timestep: int,
    section: str,
    expected: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
    *,
    atol: float,
) -> list[TraceDiff]:
    diffs: list[TraceDiff] = []
    expected_names = set(expected)
    actual_names = set(actual)

    for missing in sorted(expected_names - actual_names):
        diffs.append(
            TraceDiff(
                timestep=timestep,
                section=section,
                name=missing,
                metric="missing",
                expected=True,
                actual=False,
            )
        )
    for extra in sorted(actual_names - expected_names):
        diffs.append(
            TraceDiff(
                timestep=timestep,
                section=section,
                name=extra,
                metric="unexpected",
                expected=False,
                actual=True,
            )
        )

    for name in sorted(expected_names & actual_names):
        expected_value = expected[name]
        actual_value = actual[name]
        if expected_value.shape != actual_value.shape:
            diffs.append(
                TraceDiff(
                    timestep=timestep,
                    section=section,
                    name=name,
                    metric="shape",
                    expected=list(expected_value.shape),
                    actual=list(actual_value.shape),
                )
            )
            continue

        abs_diff = np.abs(expected_value - actual_value)
        max_abs_diff = float(abs_diff.max()) if abs_diff.size else 0.0
        if not np.allclose(expected_value, actual_value, atol=atol, rtol=0.0):
            diffs.append(
                TraceDiff(
                    timestep=timestep,
                    section=section,
                    name=name,
                    metric="values",
                    expected=expected_value.tolist(),
                    actual=actual_value.tolist(),
                    max_abs_diff=max_abs_diff,
                )
            )
    return diffs


__all__ = [
    "GoldenTrace",
    "GoldenTraceError",
    "GoldenTraceRecorder",
    "GoldenTraceValidator",
    "TRACE_SCHEMA_VERSION",
    "TraceDiff",
    "diff_traces",
    "load_golden_trace",
]
