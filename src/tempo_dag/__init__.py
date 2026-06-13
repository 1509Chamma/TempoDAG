"""Canonical public package namespace for TempoDAG."""

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

_DEVICE_EXPORTS = {
    "FPGADevice",
    "Resources",
    "Memory",
    "IO",
    "Capabilities",
    "Policies",
    "DeviceRegistry",
}
_NUMERICAL_PARITY_EXPORTS = {
    "NumericalParityConfig",
    "ONNXRuntimeParityAdapter",
    "TensorFlowKerasParityAdapter",
    "TorchQuantizedModelSimulator",
    "compare_ir_graphs",
    "run_numerical_parity_test",
}


def __getattr__(name: str) -> object:
    if name in _DEVICE_EXPORTS:
        from . import device

        return getattr(device, name)
    if name in _NUMERICAL_PARITY_EXPORTS:
        from . import numerical_parity

        return getattr(numerical_parity, name)
    raise AttributeError(f"module 'tempo_dag' has no attribute {name!r}")
