from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np

from tempo_dag.numerical_parity import _round_half_away_from_zero, quantize_array
from tempo_dag.quantization_config import (
    FixedPointSpec,
    OverflowPolicy,
    QuantizationScheme,
    QuantizationSpec,
    QuantizationType,
    StateQuantSpec,
)


@dataclass(frozen=True)
class TemporalTraceStep:
    """Per-timestep snapshot used for temporal verification and golden traces."""

    timestep: int
    inputs: dict[str, np.ndarray]
    outputs: dict[str, np.ndarray]
    state: dict[str, np.ndarray]

    def to_dict(self) -> dict[str, object]:
        return {
            "timestep": self.timestep,
            "inputs": {name: value.tolist() for name, value in self.inputs.items()},
            "outputs": {name: value.tolist() for name, value in self.outputs.items()},
            "state": {name: value.tolist() for name, value in self.state.items()},
        }


@dataclass(frozen=True)
class TemporalExecutionTrace:
    """Collection of per-timestep verification snapshots."""

    steps: tuple[TemporalTraceStep, ...]

    def to_dict(self) -> dict[str, object]:
        return {"steps": [step.to_dict() for step in self.steps]}


class TemporalParityAdapter(ABC):
    """Base class for reference runners that emit temporal execution traces."""

    @abstractmethod
    def run_sequence(self, sequence: Sequence[object]) -> TemporalExecutionTrace:
        """Run a temporal model over a sequence of timestep inputs."""


class StreamingPyTorchAdapter(TemporalParityAdapter):
    """Runs a PyTorch-style stateful model one timestep at a time."""

    def __init__(
        self,
        model: Any,
        *,
        input_name: str = "input",
        output_name: str = "output",
        state_name: str = "state",
        reset_method: str = "reset_state",
    ) -> None:
        self.model = model
        self.input_name = input_name
        self.output_name = output_name
        self.state_name = state_name
        self.reset_method = reset_method

    def run_sequence(self, sequence: Sequence[object]) -> TemporalExecutionTrace:
        reset = getattr(self.model, self.reset_method, None)
        if callable(reset):
            reset()

        eval_method = getattr(self.model, "eval", None)
        if callable(eval_method):
            eval_method()

        steps = []
        torch_module = _maybe_import_torch()
        for timestep, item in enumerate(sequence):
            prepared_input = _to_model_input(item)
            no_grad_context = (
                torch_module.no_grad() if torch_module is not None else nullcontext()
            )
            with no_grad_context:
                raw_result = self.model(prepared_input)
            outputs, state = _normalize_step_result(
                raw_result,
                output_name=self.output_name,
                state_name=self.state_name,
            )
            steps.append(
                TemporalTraceStep(
                    timestep=timestep,
                    inputs={self.input_name: _to_numpy(prepared_input)},
                    outputs=outputs,
                    state=state,
                )
            )
        return TemporalExecutionTrace(tuple(steps))


class FixedPointOracle:
    """Fixed-point temporal reference that quantizes outputs and state per step."""

    def __init__(
        self,
        *,
        output_specs: Mapping[str, QuantizationSpec] | None = None,
        state_specs: Mapping[str, StateQuantSpec] | None = None,
    ) -> None:
        self.output_specs = dict(output_specs or {})
        self.state_specs = dict(state_specs or {})

    def quantize_trace(self, trace: TemporalExecutionTrace) -> TemporalExecutionTrace:
        quantized_steps = []
        for step in trace.steps:
            quantized_outputs = {
                name: self._quantize_output(name, value)
                for name, value in step.outputs.items()
            }
            quantized_state = {
                name: self._quantize_state(name, value)
                for name, value in step.state.items()
            }
            quantized_steps.append(
                TemporalTraceStep(
                    timestep=step.timestep,
                    inputs={name: value.copy() for name, value in step.inputs.items()},
                    outputs=quantized_outputs,
                    state=quantized_state,
                )
            )
        return TemporalExecutionTrace(tuple(quantized_steps))

    def _quantize_output(self, name: str, value: np.ndarray) -> np.ndarray:
        spec = self.output_specs.get(name)
        if spec is None:
            return value.copy()
        return quantize_array(value, spec).dequantized

    def _quantize_state(self, name: str, value: np.ndarray) -> np.ndarray:
        spec = self.state_specs.get(name)
        if spec is None:
            return value.copy()

        quantized = quantize_array(value, _state_spec_to_quant_spec(spec))
        if (
            spec.overflow_policy is OverflowPolicy.ERROR
            and quantized.clipped_values > 0
        ):
            raise ValueError(
                f"state '{name}' overflowed fixed-point range during oracle execution"
            )
        if spec.overflow_policy is OverflowPolicy.WRAP and quantized.clipped_values > 0:
            return _wrap_fixed_point(value, spec)
        return quantized.dequantized


def _normalize_step_result(
    result: object,
    *,
    output_name: str,
    state_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if isinstance(result, Mapping):
        if "outputs" in result:
            outputs_raw = result.get("outputs", {})
            state_raw = result.get("state", {})
            return _normalize_named_arrays(outputs_raw), _normalize_named_arrays(
                state_raw
            )
        output_value = result.get(output_name)
        state_value = result.get(state_name, {})
        if output_value is None:
            raise ValueError("mapping result must contain an output payload")
        return {output_name: _to_numpy(output_value)}, _normalize_state_value(
            state_value, state_name=state_name
        )

    if isinstance(result, tuple) and len(result) == 2:
        output_value, state_value = result
        return {output_name: _to_numpy(output_value)}, _normalize_state_value(
            state_value, state_name=state_name
        )

    return {output_name: _to_numpy(result)}, {}


def _normalize_named_arrays(value: object) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise ValueError("expected a mapping of named arrays")
    return {str(name): _to_numpy(item) for name, item in value.items()}


def _normalize_state_value(
    state_value: object,
    *,
    state_name: str,
) -> dict[str, np.ndarray]:
    if state_value is None:
        return {}
    if isinstance(state_value, Mapping):
        return {str(name): _to_numpy(item) for name, item in state_value.items()}
    return {state_name: _to_numpy(state_value)}


def _to_model_input(value: object) -> object:
    torch = _maybe_import_torch()

    if torch is not None and isinstance(value, np.ndarray):
        return torch.as_tensor(value, dtype=torch.float32)
    return value


def _to_numpy(value: object) -> np.ndarray:
    torch = _maybe_import_torch()

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def _maybe_import_torch() -> Any | None:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return None
    return torch


def _state_spec_to_quant_spec(spec: StateQuantSpec) -> QuantizationSpec:
    fixed_point = spec.fixed_point or _infer_fixed_point_spec(spec.scale)
    return QuantizationSpec(
        bit_width=fixed_point.integer_bits + fixed_point.fractional_bits,
        scheme=QuantizationScheme.SYMMETRIC,
        qtype=QuantizationType.FIXED_POINT,
        fixed_point=fixed_point,
        scale=spec.scale,
        zero_point=spec.zero_point,
    )


def _infer_fixed_point_spec(scale: float) -> FixedPointSpec:
    fractional_bits = int(round(-np.log2(scale)))
    return FixedPointSpec(integer_bits=16, fractional_bits=max(0, fractional_bits))


def _wrap_fixed_point(value: np.ndarray, spec: StateQuantSpec) -> np.ndarray:
    quant_spec = _state_spec_to_quant_spec(spec)
    if quant_spec.fixed_point is None:
        return value.copy()

    scale = float(quant_spec.scale or 2.0 ** (-quant_spec.fixed_point.fractional_bits))
    zero_point = int(quant_spec.zero_point or 0)
    qmin = -(2 ** (quant_spec.bit_width - 1))
    qmax = (2 ** (quant_spec.bit_width - 1)) - 1
    modulus = qmax - qmin + 1

    scaled = _round_half_away_from_zero((_to_numpy(value) / scale) + zero_point).astype(
        np.int64,
        copy=False,
    )
    wrapped = ((scaled - qmin) % modulus) + qmin
    return (wrapped.astype(np.float64) - zero_point) * scale


__all__ = [
    "FixedPointOracle",
    "StreamingPyTorchAdapter",
    "TemporalExecutionTrace",
    "TemporalParityAdapter",
    "TemporalTraceStep",
]
