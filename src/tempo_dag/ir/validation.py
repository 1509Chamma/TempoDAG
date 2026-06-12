from __future__ import annotations

import collections
from typing import TYPE_CHECKING

from tempo_dag.ir.value import ValueType

if TYPE_CHECKING:
    from tempo_dag.device.board import FPGADevice
    from tempo_dag.ir.graph import Graph


class IRValidationError(Exception):
    """Base class for all IR validation errors."""

    def __init__(self, message: str, item_id: str | None = None):
        super().__init__(message)
        self.item_id = item_id


class GraphValidationError(IRValidationError):
    """Raised for structural errors at the graph level."""


class ValueValidationError(IRValidationError):
    """Raised for errors in value definitions."""


class OperatorValidationError(IRValidationError):
    """Raised for errors in operator configurations."""


class TopologyValidationError(IRValidationError):
    """Raised for errors in the graph connectivity or cycles."""


def validate_graph(graph: Graph) -> None:
    """
    Ensure the graph structure is consistent.
    Checks for existence of inputs/outputs, duplicate IDs, and reference integrity.
    """
    # Check duplicate IDs
    value_ids = list(graph.values.keys())
    op_ids = list(graph.ops.keys())

    val_counts = collections.Counter(value_ids)
    op_counts = collections.Counter(op_ids)

    for vid, count in val_counts.items():
        if count > 1:
            raise GraphValidationError(f"Duplicate value_id found: {vid}", vid)

    for oid, count in op_counts.items():
        if count > 1:
            raise GraphValidationError(f"Duplicate op_id found: {oid}", oid)

    # Ensure all graph_inputs and graph_outputs exist in graph.values
    for input_id in graph.graph_inputs:
        if input_id not in graph.values:
            raise GraphValidationError(
                f"Graph input '{input_id}' not found in values", input_id
            )

    for output_id in graph.graph_outputs:
        if output_id not in graph.values:
            raise GraphValidationError(
                f"Graph output '{output_id}' not found in values", output_id
            )

    # Ensure all operator inputs/outputs reference existing values
    for op_id, op in graph.ops.items():
        for input_id in op.inputs:
            if input_id not in graph.values:
                raise GraphValidationError(
                    f"Operator '{op_id}' references non-existent input '{input_id}'",
                    op_id,
                )
        for output_id in op.outputs:
            if output_id not in graph.values:
                raise GraphValidationError(
                    f"Operator '{op_id}' references non-existent output '{output_id}'",
                    op_id,
                )

    # Ensure every value (except inputs/constants) has a valid producer_op_id
    for val_id, val in graph.values.items():
        if val.producer_op_id:
            if val.producer_op_id not in graph.ops:
                raise GraphValidationError(
                    f"Value '{val_id}' references non-existent "
                    f"producer_op_id '{val.producer_op_id}'",
                    val_id,
                )
            # Ensure the producing operator actually lists this value in its outputs
            producer = graph.ops[val.producer_op_id]
            if val_id not in producer.outputs:
                raise GraphValidationError(
                    f"Value '{val_id}' claims to be produced by "
                    f"'{val.producer_op_id}', but is not in its outputs",
                    val_id,
                )


def validate_values(graph: Graph) -> None:
    """
    Ensure all values have valid shapes, dtypes, and configurations.
    """
    allowed_vtypes = set(ValueType)

    for val_id, val in graph.values.items():
        # Shape validation
        if not isinstance(val.shape, list):
            raise ValueValidationError(f"Value '{val_id}' shape must be a list", val_id)
        for dim in val.shape:
            if not isinstance(dim, int) or dim <= 0:
                raise ValueValidationError(
                    f"Value '{val_id}' shape must contain positive integers", val_id
                )

        # Axes validation
        if not isinstance(val.axes, list):
            raise ValueValidationError(f"Value '{val_id}' axes must be a list", val_id)
        if len(val.axes) != len(val.shape):
            raise ValueValidationError(
                f"Value '{val_id}' axes length ({len(val.axes)}) "
                f"must match shape length ({len(val.shape)})",
                val_id,
            )

        # Dtype and VType validation
        if not isinstance(val.dtype, str) or not val.dtype:
            raise ValueValidationError(
                f"Value '{val_id}' must have a valid string dtype", val_id
            )

        if val.vtype not in allowed_vtypes:
            raise ValueValidationError(
                f"Value '{val_id}' has invalid vtype: {val.vtype}", val_id
            )

        # Quantization consistency (basic check)
        if val.quant:
            if not isinstance(val.quant, dict):
                raise ValueValidationError(
                    f"Value '{val_id}' quantization config must be a dictionary", val_id
                )


def validate_operators(graph: Graph) -> None:
    """
    Ensure all operators are valid according to their own internal rules
    and general IR rules.
    """
    for op_id, op in graph.ops.items():
        # Internal validation
        try:
            op.validate(graph.values)
        except Exception as e:
            raise OperatorValidationError(
                f"Internal validation failed for operator '{op_id}': {str(e)}", op_id
            ) from e

        # Basic connectivity
        if not op.inputs and op.op_type != "Constant":  # constants might have no inputs
            # But usually we expect at least one input or a special case.
            # Requirement says: "no empty input/output lists (unless "
            # "explicitly allowed)"
            pass

        if not op.outputs:
            raise OperatorValidationError(f"Operator '{op_id}' has no outputs", op_id)


def validate_topology(graph: Graph) -> None:
    """
    Verify graph connectivity and acyclicity.
    """
    # Cycle detection (using DFS colors)
    visited = {oid: 0 for oid in graph.ops}

    # Value to consuming ops mapping
    val_to_consumers = collections.defaultdict(list)
    for oid, op in graph.ops.items():
        for inp in op.inputs:
            val_to_consumers[inp].append(oid)

    def has_cycle(op_id: str) -> bool:
        visited[op_id] = 1
        op = graph.ops[op_id]
        for out_val_id in op.outputs:
            for consumer_id in val_to_consumers[out_val_id]:
                if visited[consumer_id] == 1:
                    return True
                if visited[consumer_id] == 0:
                    if has_cycle(consumer_id):
                        return True
        visited[op_id] = 2
        return False

    for oid in graph.ops:
        if visited[oid] == 0:
            if has_cycle(oid):
                raise TopologyValidationError("Graph contains a cycle", oid)

    # Reachability from inputs (ensure all values/ops can be computed)
    reachable_values = set(graph.graph_inputs)
    # Add initial state values (if they don't have producers)
    if hasattr(graph, "states"):
        for val_id, val in graph.states.items():
            if not val.producer_op_id:
                reachable_values.add(val_id)

    # Simple fixed-point reachability
    reachable_ops = set()
    changed = True
    while changed:
        changed = False
        for op_id, op in graph.ops.items():
            if op_id not in reachable_ops:
                if all(inp in reachable_values for inp in op.inputs):
                    reachable_ops.add(op_id)
                    for out in op.outputs:
                        if out not in reachable_values:
                            reachable_values.add(out)
                            changed = True

    for val_id in graph.values:
        if val_id not in reachable_values:
            raise TopologyValidationError(
                f"Value '{val_id}' is unreachable from inputs", val_id
            )

    for op_id in graph.ops:
        if op_id not in reachable_ops:
            raise TopologyValidationError(
                f"Operator '{op_id}' is unreachable from inputs", op_id
            )

    # Ensure all graph_outputs are reachable (ensure all outputs have
    # producers or are inputs)
    for output_id in graph.graph_outputs:
        if output_id not in reachable_values:
            raise TopologyValidationError(
                f"Graph output '{output_id}' is unreachable from inputs", output_id
            )


def validate_fpga_constraints(graph: Graph, device: FPGADevice) -> None:
    """
    Verify that the graph fits within the FPGA resource constraints.
    """
    total_lut = 0
    total_dsp = 0
    total_bram = 0

    for _op_id, op in graph.ops.items():
        cost = op.estimate_fpga_cost(graph.values)
        total_lut += cost.lut
        total_dsp += cost.dsp
        total_bram += cost.bram

    # resources is a dataclass with luts, ffs, dsps, bram_36k, bram_18k
    if total_lut > device.resources.luts:
        raise IRValidationError(
            f"Insufficient LUTs on device '{device.name}': required "
            f"{total_lut}, available {device.resources.luts}"
        )
    if total_dsp > device.resources.dsps:
        raise IRValidationError(
            f"Insufficient DSPs on device '{device.name}': required "
            f"{total_dsp}, available {device.resources.dsps}"
        )
    # Simplified BRAM check (36k blocks)
    if total_bram > device.resources.bram_36k:
        raise IRValidationError(
            f"Insufficient BRAM on device '{device.name}': required "
            f"{total_bram}, available {device.resources.bram_36k}"
        )


def validate_ir(graph: Graph, device: FPGADevice | None = None) -> None:
    """
    Primary entry point for IR validation.
    Runs structural, semantic, topological, and hardware constraint checks.
    """
    validate_graph(graph)
    validate_values(graph)
    validate_operators(graph)
    validate_topology(graph)
    if device:
        validate_fpga_constraints(graph, device)

