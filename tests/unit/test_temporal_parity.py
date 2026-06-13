from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from tempo_dag.quantization_config import FixedPointSpec, OverflowPolicy, StateQuantSpec
from tempo_dag.verification.temporal_parity import (
    FixedPointOracle,
    StreamingPyTorchAdapter,
    TemporalExecutionTrace,
    TemporalTraceStep,
)


class TinyStreamingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reset_state()

    def reset_state(self) -> None:
        self.hidden = torch.zeros(2, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        self.hidden = self.hidden + x
        output = self.hidden * 0.5
        return {"output": output, "state": {"hidden": self.hidden.clone()}}


class TupleStreamingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reset_state()

    def reset_state(self) -> None:
        self.hidden = torch.tensor([1.0], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.hidden = self.hidden + x
        return self.hidden.clone(), self.hidden.clone()


def test_streaming_pytorch_adapter_traces_timestep_outputs_and_state() -> None:
    model = TinyStreamingModel()
    adapter = StreamingPyTorchAdapter(model)
    trace = adapter.run_sequence(
        [
            torch.tensor([1.0, -1.0], dtype=torch.float32),
            torch.tensor([0.5, 0.25], dtype=torch.float32),
        ]
    )

    assert isinstance(trace, TemporalExecutionTrace)
    assert len(trace.steps) == 2
    assert trace.steps[0].timestep == 0
    assert np.allclose(trace.steps[0].outputs["output"], np.array([0.5, -0.5]))
    assert np.allclose(trace.steps[1].state["hidden"], np.array([1.5, -0.75]))


def test_streaming_pytorch_adapter_normalizes_tuple_results() -> None:
    model = TupleStreamingModel()
    adapter = StreamingPyTorchAdapter(model, state_name="hidden")
    trace = adapter.run_sequence(
        [
            torch.tensor([2.0], dtype=torch.float32),
            torch.tensor([-0.5], dtype=torch.float32),
        ]
    )

    assert np.allclose(trace.steps[0].outputs["output"], np.array([3.0]))
    assert np.allclose(trace.steps[1].state["hidden"], np.array([2.5]))


def test_fixed_point_oracle_quantizes_state_when_output_specs_are_absent() -> None:
    trace = TemporalExecutionTrace(
        steps=(
            TemporalTraceStep(
                timestep=0,
                inputs={"input": np.array([0.2])},
                outputs={"output": np.array([0.375])},
                state={"hidden": np.array([0.625])},
            ),
        )
    )
    oracle = FixedPointOracle(
        output_specs={},
        state_specs={
            "hidden": StateQuantSpec(
                dtype="fixed8",
                scale=2**-4,
                overflow_policy=OverflowPolicy.SATURATE,
                fixed_point=FixedPointSpec(integer_bits=4, fractional_bits=4),
            )
        },
    )

    quantized = oracle.quantize_trace(trace)

    assert np.allclose(quantized.steps[0].outputs["output"], np.array([0.375]))
    assert np.allclose(quantized.steps[0].state["hidden"], np.array([0.625]))


def test_fixed_point_oracle_raises_on_error_overflow_policy() -> None:
    trace = TemporalExecutionTrace(
        steps=(
            TemporalTraceStep(
                timestep=0,
                inputs={"input": np.array([0.0])},
                outputs={"output": np.array([0.0])},
                state={"hidden": np.array([16.0])},
            ),
        )
    )
    oracle = FixedPointOracle(
        state_specs={
            "hidden": StateQuantSpec(
                dtype="fixed8",
                scale=2**-4,
                overflow_policy=OverflowPolicy.ERROR,
                fixed_point=FixedPointSpec(integer_bits=4, fractional_bits=4),
            )
        }
    )

    with pytest.raises(ValueError, match="overflowed fixed-point range"):
        oracle.quantize_trace(trace)


def test_fixed_point_oracle_wraps_when_requested() -> None:
    trace = TemporalExecutionTrace(
        steps=(
            TemporalTraceStep(
                timestep=0,
                inputs={"input": np.array([0.0])},
                outputs={"output": np.array([0.0])},
                state={"hidden": np.array([16.0])},
            ),
        )
    )
    oracle = FixedPointOracle(
        state_specs={
            "hidden": StateQuantSpec(
                dtype="fixed8",
                scale=2**-4,
                overflow_policy=OverflowPolicy.WRAP,
                fixed_point=FixedPointSpec(integer_bits=4, fractional_bits=4),
                zero_point=3,
            )
        }
    )

    quantized = oracle.quantize_trace(trace)

    assert np.allclose(quantized.steps[0].state["hidden"], np.array([0.0]))
