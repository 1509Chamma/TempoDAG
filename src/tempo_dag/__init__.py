"""Canonical public package namespace for TempoDAG."""

from .device import (
    IO,
    Capabilities,
    DeviceRegistry,
    FPGADevice,
    Memory,
    Policies,
    Resources,
)
from .numerical_parity import (
    NumericalParityConfig,
    ONNXRuntimeParityAdapter,
    TensorFlowKerasParityAdapter,
    TorchQuantizedModelSimulator,
    compare_ir_graphs,
    run_numerical_parity_test,
)

__version__ = "0.1.0"

__all__ = [
    "FPGADevice",
    "Resources",
    "Memory",
    "IO",
    "Capabilities",
    "Policies",
    "DeviceRegistry",
    "NumericalParityConfig",
    "ONNXRuntimeParityAdapter",
    "TensorFlowKerasParityAdapter",
    "TorchQuantizedModelSimulator",
    "__version__",
    "compare_ir_graphs",
    "run_numerical_parity_test",
]
