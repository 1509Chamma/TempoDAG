from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path

from tempo_dag.codegen.hls.generator import render_operator_hls
from tempo_dag.ir.graph import Graph
from tempo_dag.ir_temporal import (
    Kernel,
    Process,
    TemporalExecutionContract,
    TemporalSchedule,
    TemporalStorageMapping,
    derive_temporal_baseline_report,
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
    BASELINE_REPORT_JSON = "baseline_report_json"
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
    parameters: dict[str, object] | None = None,
    pipeline_ii: int = 1,
) -> str:
    """Render a top-level HLS wrapper for a temporal process.

    When `parameters` supplies values (value-id -> array), those parameters
    are BAKED into the translation unit as `static const` arrays and removed
    from the step function's port list. This keeps cosim's TV interposer
    from opening one .dat file per partitioned scalar port (Windows caps
    stdio handles at 512, which aborted GRU/LSTM cosim).
    """

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
        cpp_dtype = _to_cpp_dtype(buffer.dtype)
        if _elems(list(buffer.shape)) == 1:
            # flat [depth] so the buffer binds to `T window[Window]` kernels
            buffer_blocks.append(
                f"static {cpp_dtype} {buffer.buffer_id}[{buffer.depth}];"
            )
        else:
            shape_suffix = "".join(f"[{dim}]" for dim in buffer.shape)
            buffer_blocks.append(
                f"static {cpp_dtype} {buffer.buffer_id}"
                f"[{buffer.depth}]{shape_suffix};"
            )

    edge_delta_comments = [
        f"// edge_delta: {edge.source} -> {edge.target} lag={edge.lag_cycles}"
        for edge in process.edge_delta
    ]
    header_blocks = ["#include <cmath>", "#include <cstddef>", "#include <cstdint>"]
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
            *_render_step_function(process, kernel, pipeline_ii, parameters=parameters),
            "",
        ]
    )


class _WiringUnsupported(Exception):
    """Raised when the MVP wiring emitter cannot wire a kernel graph."""


_BINARY_ELEMENTWISE = {"Add", "Sub", "Mul", "Div"}
_UNARY_ELEMENTWISE = {"Tanh", "Sigmoid", "ReLU", "GELU", "Sqrt"}
_WIREABLE_OP_TYPES = (
    {"RollingMean", "Conv1D", "MatMul", "FusedMatMulAdd", "FusedMatMulAddActivation"}
    | _BINARY_ELEMENTWISE
    | _UNARY_ELEMENTWISE
)


def _flat_index_expr(vid: str, shape: list[int], idx: str) -> str:
    """Element expr for rank-1 [N] or rank-2 [1, N] values."""
    if len(shape) == 1:
        return f"{vid}[{idx}]"
    if len(shape) == 2 and shape[0] == 1:
        return f"{vid}[0][{idx}]"
    raise _WiringUnsupported(
        f"'{vid}' shape {shape} unsupported for element access (need [N] or [1,N])"
    )


def _shape_of(value: object) -> list[int]:
    return [int(dim) for dim in (getattr(value, "shape", None) or [])]


def _elems(shape: list[int]) -> int:
    product = 1
    for dim in shape:
        product *= dim
    return product


def _suffix(shape: list[int]) -> str:
    return "".join(f"[{dim}]" for dim in shape)


def _value_is_parameter(value: object) -> bool:
    quant = getattr(value, "quant", None)
    return getattr(value, "layout", None) == "parameter" or (
        isinstance(quant, dict) and quant.get("role") == "parameter"
    )


def _topological_ops(graph: Graph) -> list[str]:
    producers = {
        output_id: op_id
        for op_id, operator in graph.ops.items()
        for output_id in operator.outputs
    }
    indegree = {op_id: 0 for op_id in graph.ops}
    adjacency: dict[str, list[str]] = {op_id: [] for op_id in graph.ops}
    for op_id, operator in graph.ops.items():
        for input_id in operator.inputs:
            producer = producers.get(input_id)
            if producer is not None and producer != op_id:
                adjacency[producer].append(op_id)
                indegree[op_id] += 1
    ready = sorted(op_id for op_id, deg in indegree.items() if deg == 0)
    order: list[str] = []
    while ready:
        op_id = ready.pop(0)
        order.append(op_id)
        for target in sorted(adjacency[op_id]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
        ready.sort()
    if len(order) != len(graph.ops):
        raise _WiringUnsupported("kernel graph contains an operator cycle")
    return order


def _state_bindings(process: Process, kernel: Kernel) -> list[dict]:
    """Per-state (read value, write value, initializer) bindings, elems==1 only."""
    bindings = []
    for state_id, state in sorted(process.states.items()):
        read_vid = next(
            (
                e.value_id
                for e in process.edge0
                if e.source == state_id and e.target == kernel.kernel_id and e.value_id
            ),
            None,
        )
        write_vid = next(
            (
                e.value_id
                for e in process.edge_delta
                if e.source == kernel.kernel_id and e.target == state_id and e.value_id
            ),
            None,
        )
        if read_vid is None and write_vid is None:
            continue
        if read_vid is None or write_vid is None:
            raise _WiringUnsupported(
                f"state '{state_id}' needs both a read Edge0 and a write "
                f"EdgeDelta with value_ids"
            )
        elems = _elems(list(state.shape))
        init = state.metadata.get("initializer", [0.0])
        init_list = (
            [float(v) for v in init]
            if isinstance(init, (list, tuple))
            else [float(init)]
        )
        if len(init_list) not in (1, elems):
            raise _WiringUnsupported(
                f"state '{state_id}' initializer length {len(init_list)} "
                f"does not match {elems} elements"
            )
        if len(init_list) == 1 and elems > 1:
            if init_list[0] != 0.0:
                raise _WiringUnsupported(
                    f"state '{state_id}': non-zero scalar initializer for "
                    f"vector state not supported"
                )
            init_list = [0.0]
        bindings.append(
            {
                "state_id": state_id,
                "read": read_vid,
                "write": write_vid,
                "init": init_list,
                "elems": elems,
                "dtype": state.dtype,
            }
        )
    return bindings


def _step_interface(
    process: Process, kernel: Kernel
) -> tuple[list[str], list[str], list[str]]:
    graph = kernel.graph
    state_reads = {b["read"] for b in _state_bindings(process, kernel)}
    state_writes = {b["write"] for b in _state_bindings(process, kernel)}
    stream_ins = [
        vid
        for vid in graph.graph_inputs
        if not _value_is_parameter(graph.values[vid]) and vid not in state_reads
    ]
    produced = {
        output_id for operator in graph.ops.values() for output_id in operator.outputs
    }
    referenced = [
        input_id for operator in graph.ops.values() for input_id in operator.inputs
    ]
    # Parameters = declared parameter inputs PLUS referenced values that are
    # neither produced nor graph inputs (ONNX initializers land here).
    params = [
        vid for vid in graph.graph_inputs if _value_is_parameter(graph.values[vid])
    ]
    for vid in sorted(set(referenced)):
        if (
            vid not in produced
            and vid not in graph.graph_inputs
            and vid not in params
            and vid in graph.values
        ):
            params.append(vid)
    outs = [vid for vid in graph.graph_outputs if vid not in state_writes]
    if len(stream_ins) != 1 or not outs:
        raise _WiringUnsupported(
            "MVP wiring supports exactly one stream input and >=1 output"
        )
    return stream_ins, params, outs


def _baked_parameter_ids(
    params: list[str], parameters: dict[str, object] | None
) -> list[str]:
    """Parameters that get baked as in-TU `static const` arrays.

    A parameter is baked exactly when its VALUE is available in the
    `parameters` dict. Params without values stay top-level ports. This is
    the single decision point shared by the process emitter and the
    testbench emitter so their signatures always agree.
    """
    if not parameters:
        return []
    return [vid for vid in params if vid in parameters]


def _step_wiring(
    process: Process,
    kernel: Kernel,
    parameters: dict[str, object] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return (port declarations, body lines, file-scope prelude lines)."""

    graph = kernel.graph
    for operator in graph.ops.values():
        if operator.op_type not in _WIREABLE_OP_TYPES:
            raise _WiringUnsupported(
                f"unsupported op type '{operator.op_type}' for MVP wiring"
            )
    stream_ins, params, outs = _step_interface(process, kernel)
    states = _state_bindings(process, kernel)

    def _consumers(vid: str) -> list[str]:
        return [op.op_type for op in graph.ops.values() if vid in op.inputs]

    # A stream input is a scalar port only when every consumer takes scalars
    # (RollingMean); elementwise kernels need `const T x[1]` array ports.
    scalar_ids = {
        vid
        for vid in stream_ins
        if _elems(_shape_of(graph.values[vid])) == 1
        and all(c == "RollingMean" for c in _consumers(vid))
    }

    def scalar_expr(vid: str) -> str:
        shape = _shape_of(graph.values[vid])
        return vid if vid in scalar_ids else vid + "[0]" * len(shape)

    def arr1d_expr(vid: str) -> str:
        if vid in scalar_ids:
            raise _WiringUnsupported(f"'{vid}' is scalar where 1-D array expected")
        shape = _shape_of(graph.values[vid])
        return vid + "[0]" * (len(shape) - 1)

    baked = _baked_parameter_ids(params, parameters)
    baked_set = set(baked)

    # Parameter baking: params with known values become file-scope
    # `static const` arrays in the step function's translation unit instead
    # of top-level ports. Cosim's TV interposer opens one .dat file per
    # partitioned scalar port; Windows caps stdio handles at 512, which
    # aborted GRU (~1045 param elems) and LSTM (~1360) with
    # "ERROR: Error on TV file". Baked constants have no TV files at all.
    prelude: list[str] = []
    # `baked` is non-empty only when parameter VALUES were supplied, so this
    # fallback is never exercised for a baked id; it just keeps the access
    # statically non-None.
    param_values = parameters or {}
    for vid in baked:
        value = graph.values[vid]
        shape = _shape_of(value)
        prelude.append(
            f"static const {_to_cpp_dtype(value.dtype)} {vid}"
            f"{_suffix(shape)} = {_braces(param_values[vid], shape)};"
        )

    ports: list[str] = []
    for vid in stream_ins:
        value = graph.values[vid]
        dt = _to_cpp_dtype(value.dtype)
        shape = _shape_of(value)
        ports.append(
            f"const {dt} {vid}"
            if vid in scalar_ids
            else f"const {dt} {vid}{_suffix(shape)}"
        )
    for vid in params:
        if vid in baked_set:
            continue
        value = graph.values[vid]
        ports.append(
            f"const {_to_cpp_dtype(value.dtype)} {vid}{_suffix(_shape_of(value))}"
        )
    for vid in outs:
        value = graph.values[vid]
        ports.append(f"{_to_cpp_dtype(value.dtype)} {vid}{_suffix(_shape_of(value))}")

    def _partition_pragma(vid: str) -> list[str]:
        elems = _elems(_shape_of(graph.values[vid]))
        if 4 <= elems <= 256:
            return [f"#pragma HLS ARRAY_PARTITION variable={vid} " f"complete dim=0"]
        return []

    port_ids = set(stream_ins) | set(params) | set(outs)
    body: list[str] = []
    # Partition BAKED parameters so unrolled kernels are not throttled by
    # BRAM/ROM ports (the measured mechanism behind II=8 and the >1h
    # scheduling explosions: 16x16 weights over 2 ports -> II floor).
    # Top-level PORTS (stream inputs and any un-baked params) deliberately
    # get NO ARRAY_PARTITION anymore: each partitioned scalar port becomes
    # its own cosim TV .dat file and blows the Windows 512-handle stdio cap.
    for vid in baked:
        body.extend(_partition_pragma(vid))
    for binding in states:
        dt = _to_cpp_dtype(binding["dtype"])
        sid, elems = binding["state_id"], binding["elems"]
        read_vid = binding["read"]
        read_shape = _shape_of(graph.values[read_vid])
        if elems == 1:
            body.append(
                f"  static {dt} {sid}__state = "
                f"{_float_literal(binding['init'][0])};"
            )
            body.append(f"  {dt} {read_vid}{_suffix(read_shape)};")
            body.append(f"  {read_vid}{'[0]' * len(read_shape)} = {sid}__state;")
        else:
            init = (
                "{0}"
                if all(v == 0.0 for v in binding["init"])
                else "{" + ", ".join(_float_literal(v) for v in binding["init"]) + "}"
            )
            body.append(f"  static {dt} {sid}__state[{elems}] = {init};")
            body.append(f"#pragma HLS ARRAY_PARTITION variable={sid}__state complete")
            body.append(f"  {dt} {read_vid}{_suffix(read_shape)};")
            body.extend(_partition_pragma(read_vid))
            body.append(f"{sid}_read_loop:")
            body.append(f"  for (int i = 0; i < {elems}; ++i) {{")
            body.append(
                f"    {_flat_index_expr(read_vid, read_shape, 'i')} = "
                f"{sid}__state[i];"
            )
            body.append("  }")
    for operator in graph.ops.values():
        for output_id in operator.outputs:
            if output_id in port_ids:
                continue
            value = graph.values[output_id]
            body.append(
                f"  {_to_cpp_dtype(value.dtype)} "
                f"{output_id}{_suffix(_shape_of(value))};"
            )
            body.extend(_partition_pragma(output_id))

    def _arr_elem(vid):
        return _flat_index_expr(vid, _shape_of(graph.values[vid]), "i")

    _EW_ALL = _BINARY_ELEMENTWISE | _UNARY_ELEMENTWISE
    topo = _topological_ops(graph)
    run: list = []
    run_elems = None
    fuse_idx = 0

    def _flush_run():
        nonlocal run, run_elems, fuse_idx
        if len(run) >= 2:
            body.extend(
                _emit_fused_elementwise_run(graph, run, fuse_idx, port_ids, _arr_elem)
            )
            fuse_idx += 1
        elif run:
            _emit_single(run[0][0], run[0][1])
        run, run_elems = [], None

    def _emit_single(op_id, operator):
        if operator.op_type in _BINARY_ELEMENTWISE:
            lhs, rhs = operator.inputs[0], operator.inputs[1]
            body.append(
                f"  {op_id}_kernel({arr1d_expr(lhs)}, {arr1d_expr(rhs)}, "
                f"{arr1d_expr(operator.outputs[0])});"
            )
        else:
            body.append(
                f"  {op_id}_kernel({arr1d_expr(operator.inputs[0])}, "
                f"{arr1d_expr(operator.outputs[0])});"
            )

    for op_id in topo:
        operator = graph.ops[op_id]
        is_ew = operator.op_type in _EW_ALL and all(
            v not in scalar_ids for v in operator.inputs
        )
        if is_ew:
            elems = _elems(_shape_of(graph.values[operator.outputs[0]]))
            if run and elems != run_elems:
                _flush_run()
            run.append((op_id, operator))
            run_elems = elems
            continue
        _flush_run()
        if operator.op_type == "RollingMean":
            out_id = operator.outputs[0]
            dt = _to_cpp_dtype(graph.values[out_id].dtype)
            window = operator.attrs.get("window_size", 1)
            buffer_id = operator.attrs.get("buffer_id", f"{op_id}_buffer")
            out_shape = _shape_of(graph.values[out_id])
            out_ref = out_id + "[0]" * len(out_shape)
            body.append(
                f"  {op_id}_kernel<{dt}, {window}>("
                f"{scalar_expr(operator.inputs[0])}, {buffer_id}, {out_ref});"
            )
        elif operator.op_type in ("Conv1D", "MatMul"):
            data_id, weight_id = operator.inputs[0], operator.inputs[1]
            body.append(
                f"  {op_id}_kernel({data_id}, {weight_id}, " f"{operator.outputs[0]});"
            )
        elif operator.op_type in ("FusedMatMulAdd", "FusedMatMulAddActivation"):
            lhs, rhs, bias = operator.inputs[:3]
            body.append(
                f"  {op_id}_kernel({lhs}, {rhs}, {arr1d_expr(bias)}, "
                f"{operator.outputs[0]});"
            )
        elif operator.op_type in (_BINARY_ELEMENTWISE | _UNARY_ELEMENTWISE):
            _emit_single(op_id, operator)
    _flush_run()

    for binding in states:
        write_vid = binding["write"]
        write_shape = _shape_of(graph.values[write_vid])
        sid, elems = binding["state_id"], binding["elems"]
        if elems == 1:
            body.append(f"  {sid}__state = {write_vid}{'[0]' * len(write_shape)};")
        else:
            body.append(f"{sid}_write_loop:")
            body.append(f"  for (int i = 0; i < {elems}; ++i) {{")
            body.append(
                f"    {sid}__state[i] = "
                f"{_flat_index_expr(write_vid, write_shape, 'i')};"
            )
            body.append("  }")
    return ports, body, prelude


def _matmul_macs(kernel: Kernel) -> int:
    total = 0
    for op in kernel.graph.ops.values():
        if op.op_type in ("MatMul", "FusedMatMulAdd", "FusedMatMulAddActivation"):
            lhs = kernel.graph.values[op.inputs[0]]
            rhs = kernel.graph.values[op.inputs[1]]
            total += lhs.shape[0] * lhs.shape[1] * rhs.shape[1]
    return total


_EW_EXPR = {
    "Add": "({a} + {b})",
    "Sub": "({a} - {b})",
    "Mul": "({a} * {b})",
    "Div": "({a} / {b})",
    "Tanh": "std::tanh({a})",
    "Sigmoid": "(1.0f / (1.0f + std::exp(-({a}))))",
    "ReLU": "(({a}) > 0 ? ({a}) : 0)",
    "Sqrt": "std::sqrt({a})",
}


def _emit_fused_elementwise_run(graph, run, idx, port_ids, arr_expr):
    """One pipelined loop computing a chain of elementwise ops per element."""
    consumers: dict[str, list[str]] = {}
    for c_oid, op in graph.ops.items():
        for i in op.inputs:
            consumers.setdefault(i, []).append(c_oid)
    run_ids = {oid for oid, _ in run}
    out0 = run[0][1].outputs[0]
    elems = _elems(_shape_of(graph.values[out0]))
    lines = [
        f"fused_ew_{idx}_loop:",
        f"  for (int i = 0; i < {elems}; ++i) {{",
        "#pragma HLS PIPELINE II=1",
    ]
    exprs: dict[str, str] = {}

    def ref(vid):
        if vid in exprs:
            return exprs[vid]
        return arr_expr(vid)

    for _oid, op in run:
        tmpl = _EW_EXPR[op.op_type]
        expr = (
            tmpl.format(a=ref(op.inputs[0]), b=ref(op.inputs[1]))
            if op.op_type in _BINARY_ELEMENTWISE
            else tmpl.format(a=ref(op.inputs[0]))
        )
        out_id = op.outputs[0]
        tname = f"t_{out_id}"
        lines.append(f"    const float {tname} = {expr};")
        exprs[out_id] = tname
        used_outside = (
            out_id in port_ids
            or any(c not in run_ids for c in consumers.get(out_id, []))
            or out_id in graph.graph_outputs
        )
        if used_outside:
            lines.append(f"    {arr_expr(out_id)} = {tname};")
    lines.append("  }")
    return lines


def _render_step_function(
    process: Process,
    kernel: Kernel,
    pipeline_ii: int = 1,
    parameters: dict[str, object] | None = None,
) -> list[str]:
    try:
        ports, body, prelude = _step_wiring(process, kernel, parameters=parameters)
    except _WiringUnsupported as exc:
        return [
            f"void {process.process_id}_step() {{",
            "#pragma HLS DATAFLOW",
            "  // Operator invocation wiring is emitted by the next scheduler layer.",
            f"  // wiring unavailable: {exc}",
            "}",
        ]
    macs = _matmul_macs(kernel)
    if pipeline_ii != 0 and macs > 128:
        # Guardrail (measured): top-level PIPELINE fully unrolls kernels;
        # >128 MACs over ported arrays gave >1h modulo-scheduler grinds.
        pragma = [
            f"  // guardrail: top-level PIPELINE suppressed "
            f"({macs} MACs); per-kernel pipelining applies"
        ]
    else:
        pragma = [] if pipeline_ii == 0 else [f"#pragma HLS PIPELINE II={pipeline_ii}"]
    return [
        *prelude,
        *([""] if prelude else []),
        f"void {process.process_id}_step({', '.join(ports)}) {{",
        *pragma,
        *body,
        "}",
    ]


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


def _float_literal(value: float) -> str:
    text = f"{float(value):.9g}"
    if "." not in text and "e" not in text and "inf" not in text:
        text += ".0"
    return text + "f"


def _braces(values: object, shape: list[int]) -> str:
    """Nested C array initializer for a numpy-like array."""
    import numpy as np

    arr = np.asarray(values, dtype=np.float64).reshape(shape or [1])

    def fmt(sub: object) -> str:
        sub_arr = np.asarray(sub)
        if sub_arr.ndim == 0:
            return _float_literal(float(sub_arr))
        return "{" + ", ".join(fmt(item) for item in sub_arr) + "}"

    return fmt(arr)


def render_temporal_testbench(
    process: Process,
    golden_trace: GoldenTrace,
    parameters: dict[str, object] | None = None,
    tolerance: float = 1e-4,
) -> str:
    """Render a golden-trace testbench.

    When the process's step function is wired (MVP op set), the testbench
    drives real ports and, if `parameters` provides the weight values,
    ASSERTS every output against the golden trace (nonzero exit on
    mismatch). Otherwise it falls back to the legacy comment-only scaffold.

    Parameters present in `parameters` are BAKED into the DUT translation
    unit by the process emitter (see _step_wiring), so the testbench neither
    declares nor passes those arrays - call args are stream input, any
    un-baked params, then outputs. The baking decision is shared via
    _baked_parameter_ids so both sides always agree.
    """
    import numpy as np

    kernel = next(iter(process.kernels.values())) if process.kernels else None
    wired = None
    if kernel is not None:
        try:
            _step_wiring(process, kernel, parameters=parameters)
            wired = _step_interface(process, kernel)
        except _WiringUnsupported:
            wired = None

    lines = [
        f"// Testbench for {process.process_id}",
        "#include <cmath>",
        "#include <cstddef>",
        "#include <iostream>",
        "",
    ]

    if wired is None:
        lines.extend(
            [
                f"extern void {process.process_id}_step();",
                "",
                "int main() {",
                f"  const std::size_t num_steps = {len(golden_trace.steps)};",
                '  std::cout << "Running temporal golden trace (" << num_steps',
                '            << " steps)" << std::endl;',
            ]
        )
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

    # `wired` is only set when a kernel was found and wiring succeeded, so
    # `kernel` is non-None past the early return above.
    assert kernel is not None
    graph = kernel.graph
    stream_ins, params, outs = wired
    ports, _, _ = _step_wiring(process, kernel, parameters=parameters)
    baked = set(_baked_parameter_ids(params, parameters))
    port_params = [vid for vid in params if vid not in baked]
    lines.extend(
        [
            f"extern void {process.process_id}_step({', '.join(ports)});",
            "",
            "int main() {",
            f"  const std::size_t num_steps = {len(golden_trace.steps)};",
            '  std::cout << "Running temporal golden trace (" << num_steps',
            '            << " steps)" << std::endl;',
            "  int errors = 0;",
        ]
    )
    checking = parameters is not None
    # Baked params live inside the DUT translation unit; only un-baked
    # params are declared and passed here.
    for vid in port_params:
        value = graph.values[vid]
        shape = _shape_of(value)
        data = (parameters or {}).get(vid, np.zeros(shape or [1]))
        lines.append(
            f"  {_to_cpp_dtype(value.dtype)} {vid}{_suffix(shape)} = "
            f"{_braces(data, shape)};"
        )
    for vid in outs:
        value = graph.values[vid]
        lines.append(
            f"  {_to_cpp_dtype(value.dtype)} {vid}{_suffix(_shape_of(value))};"
        )
    if not checking:
        lines.append("  // parameter values not supplied: outputs not asserted")

    stream_vid = stream_ins[0]
    stream_shape = _shape_of(graph.values[stream_vid])
    stream_scalar = _elems(stream_shape) == 1 and all(
        op.op_type == "RollingMean"
        for op in graph.ops.values()
        if stream_vid in op.inputs
    )
    if not stream_scalar:
        lines.append(
            f"  {_to_cpp_dtype(graph.values[stream_vid].dtype)} "
            f"{stream_vid}{_suffix(stream_shape)};"
        )
    for step in golden_trace.steps:
        lines.append(f"  // timestep {step.timestep}")
        step_inputs = np.asarray(list(step.inputs.values())[0]).flatten()
        for name, values in step.inputs.items():
            lines.append(f"  // input {name} = {values.tolist()}")
        for name, values in step.outputs.items():
            lines.append(f"  // expected output {name} = {values.tolist()}")
        if stream_scalar:
            call_args = [_float_literal(float(step_inputs[0])), *port_params, *outs]
        else:
            for j in range(_elems(stream_shape)):
                lines.append(
                    f"  {_flat_index_expr(stream_vid, stream_shape, str(j))} "
                    f"= {_float_literal(float(step_inputs[j]))};"
                )
            call_args = [stream_vid, *port_params, *outs]
        lines.append(f"  {process.process_id}_step({', '.join(call_args)});")
        if checking and step.outputs:
            expected = float(np.asarray(list(step.outputs.values())[0]).flat[0])
            out_vid = outs[0]
            out_ref = out_vid + "[0]" * len(_shape_of(graph.values[out_vid]))
            lines.extend(
                [
                    f"  if (std::fabs({out_ref} - "
                    f"({_float_literal(expected)})) > {tolerance:g}f) {{",
                    f'    std::cerr << "MISMATCH t={step.timestep} got " '
                    f'<< {out_ref} << " want {expected:.9g}" << std::endl;',
                    "    ++errors;",
                    "  }",
                ]
            )
    lines.extend(
        [
            '  std::cout << "Temporal testbench complete, errors=" << errors',
            "            << std::endl;",
            "  return errors == 0 ? 0 : 1;",
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
    parameters: dict[str, object] | None = None,
    pipeline_ii: int = 1,
    tolerance: float = 1e-4,
) -> TemporalHLSArtifact:
    return TemporalHLSArtifact(
        process_hls=render_temporal_process_hls(
            process,
            contract=contract,
            schedule=schedule,
            parameters=parameters,
            pipeline_ii=pipeline_ii,
        ),
        testbench_hls=render_temporal_testbench(
            process,
            golden_trace,
            parameters=parameters,
            tolerance=tolerance,
        ),
    )


def write_temporal_hls_artifact_bundle(
    process: Process,
    golden_trace: GoldenTrace,
    output_dir: str | PathLike[str],
    *,
    stem: str | None = None,
    parameters: dict[str, object] | None = None,
    pipeline_ii: int = 1,
    tolerance: float = 1e-4,
) -> TemporalHLSArtifactManifest:
    """Write process, trace, HLS, testbench, and manifest artifacts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_stem = stem or process.process_id
    contract = derive_temporal_execution_contract(process)
    schedule = derive_temporal_schedule(process, contract)
    baseline_report = derive_temporal_baseline_report(
        process,
        schedule,
        trace_metadata=golden_trace.metadata,
    )
    rendered = render_temporal_artifact_from_trace(
        process,
        golden_trace,
        contract=contract,
        schedule=schedule,
        parameters=parameters,
        pipeline_ii=pipeline_ii,
        tolerance=tolerance,
    )

    process_path = output_path / f"{artifact_stem}_process.json"
    trace_path = output_path / f"{artifact_stem}_trace.json"
    schedule_path = output_path / f"{artifact_stem}_schedule.json"
    report_path = output_path / f"{artifact_stem}_baseline_report.json"
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
    report_path.write_text(
        json.dumps(
            baseline_report.to_dict(),
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
            TemporalArtifactFile(
                TemporalArtifactKind.BASELINE_REPORT_JSON,
                report_path,
            ),
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
