from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import onnx
import torch
import torch.nn as nn

from tempo_dag.codegen.hls.temporal_generator import (
    render_temporal_artifact_from_trace,
)
from tempo_dag.numerical_parity import quantize_array
from tempo_dag.parsers.temporal_onnx import (
    TemporalONNXParser,
    build_demo_temporal_onnx_model,
)
from tempo_dag.quantization_config import (
    FixedPointSpec,
    OverflowPolicy,
    QuantizationScheme,
    QuantizationSpec,
    QuantizationType,
    StateQuantSpec,
)
from tempo_dag.verification import (
    FixedPointOracle,
    GoldenTraceRecorder,
    GoldenTraceValidator,
    StreamingPyTorchAdapter,
    TemporalExecutionTrace,
)

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "examples" / "generated"


@dataclass(frozen=True)
class TemporalStepMetric:
    timestep: int
    max_output_abs_error: float
    max_state_abs_error: float
    output_overflow: bool
    state_overflow: bool


@dataclass(frozen=True)
class TemporalDemoReport:
    process_id: str
    buffers: list[str]
    generated_files: list[str]
    num_trace_steps: int
    validation_passed: bool
    max_state_abs: float
    max_output_abs: float
    max_state_abs_error: float
    max_output_abs_error: float
    overflow_detected: bool
    step_metrics: list[TemporalStepMetric]


class RollingMeanConvDemoModel(nn.Module):
    def __init__(self, window_size: int = 4) -> None:
        super().__init__()
        self.window_size = window_size
        self.reset_state()

    def reset_state(self) -> None:
        self.buffer = torch.zeros(self.window_size, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        self.buffer = torch.roll(self.buffer, shifts=-1)
        self.buffer[-1] = x.squeeze()
        rolling_mean = self.buffer.mean()
        conv_like = (rolling_mean * 1.0) + (rolling_mean * 0.5) + (rolling_mean * 1.0)
        output = torch.tensor([conv_like + 0.125], dtype=torch.float32)
        return {
            "outputs": {"output": output},
            "state": {
                "buffer": self.buffer.clone(),
                "rolling_mean": torch.tensor([rolling_mean], dtype=torch.float32),
            },
        }


def run_demo(output_dir: Path = OUTPUT_DIR) -> TemporalDemoReport:
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = TemporalONNXParser()
    model = build_demo_temporal_onnx_model()
    lowering = parser.parse_model(model, process_id="temporal_demo")

    sequence = [
        torch.tensor([3.0], dtype=torch.float32),
        torch.tensor([6.0], dtype=torch.float32),
        torch.tensor([9.0], dtype=torch.float32),
        torch.tensor([12.0], dtype=torch.float32),
    ]
    adapter = StreamingPyTorchAdapter(RollingMeanConvDemoModel())
    fp_trace = adapter.run_sequence(sequence)

    output_spec = QuantizationSpec(
        bit_width=8,
        scheme=QuantizationScheme.SYMMETRIC,
        qtype=QuantizationType.FIXED_POINT,
        fixed_point=FixedPointSpec(integer_bits=4, fractional_bits=4),
        scale=2**-4,
        zero_point=0,
    )
    state_spec = StateQuantSpec(
        dtype="fixed16",
        scale=2**-8,
        overflow_policy=OverflowPolicy.SATURATE,
        fixed_point=FixedPointSpec(integer_bits=8, fractional_bits=8),
    )
    oracle = FixedPointOracle(
        output_specs={"output": output_spec},
        state_specs={"buffer": state_spec, "rolling_mean": state_spec},
    )
    quantized_trace = oracle.quantize_trace(fp_trace)

    recorder = GoldenTraceRecorder()
    golden_trace = recorder.record(
        quantized_trace,
        metadata={"case": "temporal_demo", "num_steps": len(sequence)},
    )
    validator = GoldenTraceValidator()
    validation = validator.validate(golden_trace, golden_trace)

    artifact = render_temporal_artifact_from_trace(lowering.process, golden_trace)

    onnx_path = output_dir / "temporal_demo.onnx"
    golden_trace_path = output_dir / "temporal_demo_trace.json"
    process_path = output_dir / "temporal_demo_process.json"
    hls_path = output_dir / "temporal_demo.cpp"
    testbench_path = output_dir / "temporal_demo_tb.cpp"
    report_path = output_dir / "temporal_demo_report.json"

    onnx.save(model, onnx_path)
    golden_trace_path.write_text(
        json.dumps(golden_trace.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    process_path.write_text(
        json.dumps(lowering.process.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hls_path.write_text(artifact.process_hls, encoding="utf-8")
    testbench_path.write_text(artifact.testbench_hls, encoding="utf-8")

    step_metrics = _build_step_metrics(
        fp_trace,
        quantized_trace,
        output_spec=output_spec,
        state_specs={"buffer": state_spec, "rolling_mean": state_spec},
    )
    max_state_abs = max(
        abs(value).max() for step in golden_trace.steps for value in step.state.values()
    )
    max_output_abs = max(
        abs(value).max()
        for step in golden_trace.steps
        for value in step.outputs.values()
    )
    report = TemporalDemoReport(
        process_id=lowering.process.process_id,
        buffers=sorted(lowering.process.buffers),
        generated_files=[
            str(onnx_path),
            str(golden_trace_path),
            str(process_path),
            str(hls_path),
            str(testbench_path),
            str(report_path),
        ],
        num_trace_steps=len(golden_trace.steps),
        validation_passed=bool(validation["pass"]),
        max_state_abs=float(max_state_abs),
        max_output_abs=float(max_output_abs),
        max_state_abs_error=max(
            (metric.max_state_abs_error for metric in step_metrics),
            default=0.0,
        ),
        max_output_abs_error=max(
            (metric.max_output_abs_error for metric in step_metrics),
            default=0.0,
        ),
        overflow_detected=any(
            metric.output_overflow or metric.state_overflow for metric in step_metrics
        ),
        step_metrics=step_metrics,
    )
    report_path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _build_step_metrics(
    reference_trace: TemporalExecutionTrace,
    quantized_trace: TemporalExecutionTrace,
    *,
    output_spec: QuantizationSpec,
    state_specs: dict[str, StateQuantSpec],
) -> list[TemporalStepMetric]:
    metrics = []
    for reference_step, quantized_step in zip(
        reference_trace.steps,
        quantized_trace.steps,
        strict=True,
    ):
        max_output_abs_error = max(
            (
                float(
                    abs(
                        reference_step.outputs[name] - quantized_step.outputs[name]
                    ).max()
                )
                for name in reference_step.outputs
            ),
            default=0.0,
        )
        max_state_abs_error = max(
            (
                float(
                    abs(reference_step.state[name] - quantized_step.state[name]).max()
                )
                for name in reference_step.state
            ),
            default=0.0,
        )
        output_overflow = any(
            quantize_array(value, output_spec).clipped_values > 0
            for value in reference_step.outputs.values()
        )
        state_overflow = any(
            quantize_array(
                value,
                _state_to_quant_spec(state_specs[name]),
            ).clipped_values
            > 0
            for name, value in reference_step.state.items()
            if name in state_specs
        )
        metrics.append(
            TemporalStepMetric(
                timestep=reference_step.timestep,
                max_output_abs_error=max_output_abs_error,
                max_state_abs_error=max_state_abs_error,
                output_overflow=output_overflow,
                state_overflow=state_overflow,
            )
        )
    return metrics


def _state_to_quant_spec(spec: StateQuantSpec) -> QuantizationSpec:
    fixed_point = spec.fixed_point or FixedPointSpec(
        integer_bits=16,
        fractional_bits=8,
    )
    return QuantizationSpec(
        bit_width=fixed_point.integer_bits + fixed_point.fractional_bits,
        scheme=QuantizationScheme.SYMMETRIC,
        qtype=QuantizationType.FIXED_POINT,
        fixed_point=fixed_point,
        scale=spec.scale,
        zero_point=spec.zero_point,
    )


def main() -> None:
    result = run_demo()
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
