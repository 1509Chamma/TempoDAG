"""Compatibility wrapper for the older public IR validation module."""

from tempo_dag.ir.validation import (
    GraphValidationError,
    IRValidationError,
    OperatorValidationError,
    TopologyValidationError,
    ValueValidationError,
    validate_fpga_constraints,
    validate_graph,
    validate_ir,
    validate_operators,
    validate_topology,
    validate_values,
)

__all__ = [
    "GraphValidationError",
    "IRValidationError",
    "OperatorValidationError",
    "TopologyValidationError",
    "ValueValidationError",
    "validate_fpga_constraints",
    "validate_graph",
    "validate_ir",
    "validate_operators",
    "validate_topology",
    "validate_values",
]

