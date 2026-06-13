from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from tempo_dag.codegen.hls.generator import render_operator_hls
from tempo_dag.ir_temporal import Process
from tempo_dag.verification.golden_trace import GoldenTrace, load_golden_trace


@dataclass(frozen=True)
class TemporalHLSArtifact:
    """Rendered temporal HLS bundle for a Process."""

    process_hls: str
    testbench_hls: str


def render_temporal_process_hls(process: Process) -> str:
    """Render a top-level HLS wrapper for a temporal process."""

    process.validate()
    if len(process.kernels) != 1:
        raise ValueError("temporal HLS MVP currently supports exactly one kernel")

    kernel = next(iter(process.kernels.values()))
    operator_blocks = []
    for operator in kernel.graph.ops.values():
        operator_blocks.append(render_operator_hls(operator, kernel.graph.values))

    buffer_blocks = []
    for buffer in process.buffers.values():
        shape_suffix = "".join(f"[{dim}]" for dim in buffer.shape)
        cpp_dtype = _to_cpp_dtype(buffer.dtype)
        buffer_blocks.append(
            f"static {cpp_dtype} {buffer.buffer_id}[{buffer.depth}]{shape_suffix};"
        )

    edge_delta_comments = [
        f"// edge_delta: {edge.source} -> {edge.target} lag={edge.lag_cycles}"
        for edge in process.edge_delta
    ]

    return "\n".join(
        [
            f"// Temporal process: {process.process_id}",
            "#include <cstddef>",
            "#include <cstdint>",
            "",
            *buffer_blocks,
            "",
            *edge_delta_comments,
            "",
            f"void {process.process_id}_step() {{",
            *[f"  {line}" for block in operator_blocks for line in block.splitlines()],
            "}",
            "",
        ]
    )


def render_temporal_testbench(
    process: Process,
    golden_trace: GoldenTrace,
) -> str:
    """Render a minimal C++ testbench that replays a golden trace."""

    lines = [
        f"// Testbench for {process.process_id}",
        "#include <cmath>",
        "#include <cstddef>",
        "#include <iostream>",
        "",
        f"extern void {process.process_id}_step();",
        "",
        "int main() {",
        f"  const std::size_t num_steps = {len(golden_trace.steps)};",
        '  std::cout << "Running temporal golden trace" << std::endl;',
    ]

    for step in golden_trace.steps:
        lines.append(f"  // timestep {step.timestep}")
        for name, values in step.inputs.items():
            lines.append(f"  // input {name} = {values.tolist()}")
        for name, values in step.outputs.items():
            lines.append(f"  // expected output {name} = {values.tolist()}")
        for name, values in step.state.items():
            lines.append(f"  // expected state {name} = {values.tolist()}")
        lines.append(f"  {process.process_id}_step();")
    lines.extend(
        [
            '  std::cout << "Temporal testbench complete" << std::endl;',
            "  return 0;",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _to_cpp_dtype(dtype: str) -> str:
    return {
        "float16": "half",
        "float32": "float",
        "float64": "double",
        "int16": "std::int16_t",
        "int32": "std::int32_t",
        "int64": "std::int64_t",
    }.get(dtype, dtype)


def render_temporal_artifact_from_trace(
    process: Process,
    golden_trace: GoldenTrace,
) -> TemporalHLSArtifact:
    return TemporalHLSArtifact(
        process_hls=render_temporal_process_hls(process),
        testbench_hls=render_temporal_testbench(process, golden_trace),
    )


def load_and_render_temporal_artifact(
    process: Process,
    golden_trace_path: str | PathLike[str],
) -> TemporalHLSArtifact:
    return render_temporal_artifact_from_trace(
        process,
        load_golden_trace(Path(golden_trace_path)),
    )


__all__ = [
    "TemporalHLSArtifact",
    "load_and_render_temporal_artifact",
    "render_temporal_artifact_from_trace",
    "render_temporal_process_hls",
    "render_temporal_testbench",
]
