"""Compatibility wrapper for the older public IR registry module."""

from tempo_dag.ir.registry import (
    DuplicateOperatorError,
    OperatorRegistry,
    OperatorRegistryError,
    UnknownOperatorError,
    get_default_registry,
)

__all__ = [
    "DuplicateOperatorError",
    "OperatorRegistry",
    "OperatorRegistryError",
    "UnknownOperatorError",
    "get_default_registry",
]
