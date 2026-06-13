from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn

from tempo_dag.quantization_config import (
    FixedPointSpec,
    OverflowPolicy,
    QuantizationScheme,
    QuantizationSpec,
    QuantizationType,
    StateQuantSpec,
)
from tempo_dag.verification import (
    GoldenTraceRecorder,
    GoldenTraceValidator,
    StreamingPyTorchAdapter,
    load_golden_trace,
)

GOLDEN_TRACE_DIR = Path(__file__).with_name("golden_traces")

OUTPUT_SPEC = QuantizationSpec(
    bit_width=8,
    scheme=QuantizationScheme.SYMMETRIC,
    qtype=QuantizationType.FIXED_POINT,
    fixed_point=FixedPointSpec(integer_bits=4, fractional_bits=4),
    scale=2**-4,
    zero_point=0,
)

STATE_SPEC = StateQuantSpec(
    dtype="fixed16",
    scale=2**-4,
    overflow_policy=OverflowPolicy.SATURATE,
    fixed_point=FixedPointSpec(integer_bits=8, fractional_bits=8),
)


class DelayChainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reset_state()

    def reset_state(self) -> None:
        self.prev1 = torch.tensor([0.0], dtype=torch.float32)
        self.prev2 = torch.tensor([0.0], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        output = self.prev2.clone()
        self.prev2 = self.prev1.clone()
        self.prev1 = x.clone()
        return {
            "outputs": {"output": output},
            "state": {"prev1": self.prev1.clone(), "prev2": self.prev2.clone()},
        }


class RollingMeanModel(nn.Module):
    def __init__(self, window_size: int = 4) -> None:
        super().__init__()
        self.window_size = window_size
        self.reset_state()

    def reset_state(self) -> None:
        self.buffer = torch.zeros(self.window_size, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        self.buffer = torch.roll(self.buffer, shifts=-1)
        self.buffer[-1] = x.squeeze()
        mean = self.buffer.mean().unsqueeze(0)
        return {
            "outputs": {"output": mean},
            "state": {"buffer": self.buffer.clone()},
        }


class RunningAccumulatorModel(nn.Module):
    def __init__(self, weight: float = 0.5) -> None:
        super().__init__()
        self.weight = weight
        self.reset_state()

    def reset_state(self) -> None:
        self.acc = torch.tensor([0.0], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        self.acc = self.acc + (x * self.weight)
        return {
            "outputs": {"output": self.acc.clone()},
            "state": {"acc": self.acc.clone()},
        }


class SmallGRUModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reset_state()

    def reset_state(self) -> None:
        self.hidden = torch.tensor([0.0], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        z = torch.sigmoid((0.5 * x) + (0.25 * self.hidden))
        n = torch.tanh((0.75 * x) + (0.1 * self.hidden))
        self.hidden = ((1.0 - z) * n) + (z * self.hidden)
        return {
            "outputs": {"output": self.hidden.clone()},
            "state": {"hidden": self.hidden.clone()},
        }


class HybridRollingLinearModel(nn.Module):
    def __init__(self, window_size: int = 3) -> None:
        super().__init__()
        self.window_size = window_size
        self.reset_state()

    def reset_state(self) -> None:
        self.buffer = torch.zeros(self.window_size, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        self.buffer = torch.roll(self.buffer, shifts=-1)
        self.buffer[-1] = x.squeeze()
        rolling_mean = self.buffer.mean()
        output = (rolling_mean * 2.0).unsqueeze(0)
        return {
            "outputs": {"output": output},
            "state": {
                "buffer": self.buffer.clone(),
                "rolling_mean": rolling_mean.unsqueeze(0),
            },
        }


def _state_specs(*names: str) -> dict[str, StateQuantSpec]:
    return {name: STATE_SPEC for name in names}


def _run_case(
    case_name: str,
    model: nn.Module,
    sequence: list[torch.Tensor],
    *,
    state_specs: dict[str, StateQuantSpec],
) -> None:
    adapter = StreamingPyTorchAdapter(model)
    trace = adapter.run_sequence(sequence)

    from tempo_dag.verification import FixedPointOracle

    oracle = FixedPointOracle(
        output_specs={"output": OUTPUT_SPEC},
        state_specs=state_specs,
    )
    quantized_trace = oracle.quantize_trace(trace)
    recorder = GoldenTraceRecorder()
    recorded = recorder.record(
        quantized_trace,
        metadata={"case": case_name, "num_steps": len(sequence)},
    )
    golden_trace = load_golden_trace(GOLDEN_TRACE_DIR / f"{case_name}.json")
    validator = GoldenTraceValidator()
    report = validator.validate(golden_trace, recorded)

    assert report["pass"] is True, report["diffs"]
    assert recorded.metadata["case"] == case_name

    payload = recorded.to_dict()
    expected_payload = json.loads(
        (GOLDEN_TRACE_DIR / f"{case_name}.json").read_text(encoding="utf-8")
    )
    assert payload == expected_payload


def test_delay_chain_matches_golden_trace() -> None:
    _run_case(
        "delay_chain",
        DelayChainModel(),
        [
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([2.0], dtype=torch.float32),
            torch.tensor([3.0], dtype=torch.float32),
            torch.tensor([4.0], dtype=torch.float32),
        ],
        state_specs=_state_specs("prev1", "prev2"),
    )


def test_rolling_mean_matches_golden_trace() -> None:
    _run_case(
        "rolling_mean",
        RollingMeanModel(window_size=4),
        [
            torch.tensor([4.0], dtype=torch.float32),
            torch.tensor([8.0], dtype=torch.float32),
            torch.tensor([12.0], dtype=torch.float32),
            torch.tensor([16.0], dtype=torch.float32),
        ],
        state_specs=_state_specs("buffer"),
    )


def test_running_accumulator_matches_golden_trace() -> None:
    _run_case(
        "running_accumulator",
        RunningAccumulatorModel(weight=0.5),
        [
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([2.0], dtype=torch.float32),
            torch.tensor([-1.0], dtype=torch.float32),
            torch.tensor([4.0], dtype=torch.float32),
        ],
        state_specs=_state_specs("acc"),
    )


def test_small_gru_matches_golden_trace() -> None:
    _run_case(
        "small_gru",
        SmallGRUModel(),
        [
            torch.tensor([0.2], dtype=torch.float32),
            torch.tensor([-0.1], dtype=torch.float32),
            torch.tensor([0.5], dtype=torch.float32),
        ],
        state_specs=_state_specs("hidden"),
    )


def test_hybrid_rolling_linear_matches_golden_trace() -> None:
    _run_case(
        "hybrid_rolling_linear",
        HybridRollingLinearModel(window_size=3),
        [
            torch.tensor([3.0], dtype=torch.float32),
            torch.tensor([6.0], dtype=torch.float32),
            torch.tensor([9.0], dtype=torch.float32),
            torch.tensor([12.0], dtype=torch.float32),
        ],
        state_specs=_state_specs("buffer", "rolling_mean"),
    )


def test_validator_reports_trace_mismatch() -> None:
    golden_trace = load_golden_trace(GOLDEN_TRACE_DIR / "delay_chain.json")
    candidate = json.loads(
        (GOLDEN_TRACE_DIR / "delay_chain.json").read_text(encoding="utf-8")
    )
    candidate["steps"][2]["outputs"]["output"] = [99.0]

    report = GoldenTraceValidator().validate(golden_trace, candidate)
    diffs = cast(list[dict[str, object]], report["diffs"])

    assert report["pass"] is False
    assert diffs[0]["section"] == "outputs"
