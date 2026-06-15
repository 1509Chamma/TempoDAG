from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path

from tempo_dag.codegen.hls.generator import render_operator_hls
from tempo_dag.ir_temporal import (
    Process,
    TemporalExecutionContract,
    TemporalSchedule,
    TemporalStorageMapping,
    derive_temporal_execution_contract,
    derive_temporal_schedule,
)
from tempo_dag.verification.golden_trace import GoldenTrace, load_golden_trace


class TemporalArtifactKind(Enum):
    """Known graph-level temporal artifact categories."""

    PROCESS_JSON = "process_json"
    GOLDEN_TRACE_JSON = "golden_trace_json"
    PROCESS_HLS = "process_hls"
    TESTBENCH_HLS = "testbench_hls"
    SCHEDULE_JSON = "schedule_json"
    MANIFEST_JSON = "manifest_json"


@dataclass(frozen=True)
class TemporalArtifactFile:
    """One file emitted for a graph-level temporal HLS bundle."""

    kind: TemporalArtifactKind
    path: Path

    def to_dict(self, root: Path) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "path": self.path.relative_to(root).as_posix(),
        }


@dataclass(frozen=True)
class TemporalHLSArtifact:
    """Rendered temporal HLS bundle for a Process."""

    process_hls: str
    testbench_hls: str


@dataclass(frozen=True)
class TemporalHLSArtifactManifest:
    """Manifest for a graph-level temporal HLS artifact bundle."""

    process_id: str
    files: tuple[TemporalArtifactFile, ...]

    def to_dict(self, root: Path) -> dict[str, object]:
        return {
            "process_id": self.process_id,
            "files": [artifact.to_dict(root) for artifact in self.files],
        }


def render_temporal_process_hls(
    process: Process,
    *,
    contract: TemporalExecutionContract | None = None,
    schedule: TemporalSchedule | None = None,
) -> str:
    """Render a top-level HLS wrapper for a temporal process."""

    if contract is None:
        contract = derive_temporal_execution_contract(process)
    if schedule is None:
        schedule = derive_temporal_schedule(process, contract)
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
    header_blocks = ["#include <cstddef>", "#include <cstdint>"]
    if any(
        _to_cpp_dtype(buffer.dtype) == "half" for buffer in process.buffers.values()
    ):
        header_blocks.append("#include <hls_half.h>")

    return "\n".join(
        [
            f"// Temporal process: {process.process_id}",
            f"// reset_policy: {contract.reset_policy.value}",
            f"// warmup_timesteps: {contract.warmup_timesteps}",
            f"// flush_cycles: {contract.flush_cycles}",
            f"// schedule_estimated_latency_cycles: "
            f"{schedule.estimated_latency_cycles}",
            f"// schedule_estimated_initiation_interval: "
            f"{schedule.estimated_initiation_interval}",
            *_contract_storage_comments(
                contract.edge_delta_storage,
                "edge_delta_storage",
            ),
            *_contract_storage_comments(contract.buffer_storage, "buffer_storage"),
            *[
                f"// schedule_node: {node.node_id} kind={node.kind.value} "
                f"phase={node.phase} latency={node.latency_cycles} "
                f"ii={node.initiation_interval}"
                for node in schedule.nodes
            ],
            *[
                f"// schedule_edge: {edge.edge_id} kind={edge.kind.value} "
                f"phase={edge.phase} storage="
                f"{edge.storage_kind.value if edge.storage_kind else 'none'}"
                for edge in schedule.edges
            ],
            *header_blocks,
            "",
            *buffer_blocks,
            "",
            *edge_delta_comments,
            "",
            *[line for block in operator_blocks for line in block.splitlines()],
            "",
            f"void {process.process_id}_step() {{",
            "#pragma HLS DATAFLOW",
            "  // Operator invocation wiring is emitted by the next scheduler layer.",
            "}",
            "",
        ]
    )


def _contract_storage_comments(
    storage: tuple[TemporalStorageMapping, ...],
    label: str,
) -> list[str]:
    comments = []
    for mapping in storage:
        comments.append(
            f"// {label}: {mapping.component_id} -> "
            f"{mapping.storage_kind.value} latency={mapping.latency_cycles}"
        )
    return comments


def render_temporal_testbench(
    process: Process,
    golden_trace: GoldenTrace,
) -> str:
    """Render a minimal C++ scaffold annotated with golden-trace comments."""

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
        '  std::cout << "Running temporal golden trace (" << num_steps',
        '            << " steps)" << std::endl;',
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
    *,
    contract: TemporalExecutionContract | None = None,
    schedule: TemporalSchedule | None = None,
) -> TemporalHLSArtifact:
    return TemporalHLSArtifact(
        process_hls=render_temporal_process_hls(
            process,
            contract=contract,
            schedule=schedule,
        ),
        testbench_hls=render_temporal_testbench(process, golden_trace),
    )


def write_temporal_hls_artifact_bundle(
    process: Process,
    golden_trace: GoldenTrace,
    output_dir: str | PathLike[str],
    *,
    stem: str | None = None,
) -> TemporalHLSArtifactManifest:
    """Write process, trace, HLS, testbench, and manifest artifacts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_stem = stem or process.process_id
    contract = derive_temporal_execution_contract(process)
    schedule = derive_temporal_schedule(process, contract)
    rendered = render_temporal_artifact_from_trace(
        process,
        golden_trace,
        contract=contract,
        schedule=schedule,
    )

    process_path = output_path / f"{artifact_stem}_process.json"
    trace_path = output_path / f"{artifact_stem}_trace.json"
    schedule_path = output_path / f"{artifact_stem}_schedule.json"
    hls_path = output_path / f"{artifact_stem}.cpp"
    testbench_path = output_path / f"{artifact_stem}_tb.cpp"
    manifest_path = output_path / f"{artifact_stem}_manifest.json"

    process_path.write_text(
        json.dumps(process.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trace_path.write_text(
        json.dumps(golden_trace.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    schedule_path.write_text(
        json.dumps(
            schedule.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    hls_path.write_text(rendered.process_hls, encoding="utf-8")
    testbench_path.write_text(rendered.testbench_hls, encoding="utf-8")

    manifest = TemporalHLSArtifactManifest(
        process_id=process.process_id,
        files=(
            TemporalArtifactFile(TemporalArtifactKind.PROCESS_JSON, process_path),
            TemporalArtifactFile(TemporalArtifactKind.GOLDEN_TRACE_JSON, trace_path),
            TemporalArtifactFile(TemporalArtifactKind.SCHEDULE_JSON, schedule_path),
            TemporalArtifactFile(TemporalArtifactKind.PROCESS_HLS, hls_path),
            TemporalArtifactFile(TemporalArtifactKind.TESTBENCH_HLS, testbench_path),
            TemporalArtifactFile(TemporalArtifactKind.MANIFEST_JSON, manifest_path),
        ),
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(output_path), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_and_render_temporal_artifact(
    process: Process,
    golden_trace_path: str | PathLike[str],
) -> TemporalHLSArtifact:
    return render_temporal_artifact_from_trace(
        process,
        load_golden_trace(Path(golden_trace_path)),
    )


__all__ = [
    "TemporalArtifactFile",
    "TemporalArtifactKind",
    "TemporalHLSArtifact",
    "TemporalHLSArtifactManifest",
    "load_and_render_temporal_artifact",
    "render_temporal_artifact_from_trace",
    "render_temporal_process_hls",
    "render_temporal_testbench",
    "write_temporal_hls_artifact_bundle",
]
