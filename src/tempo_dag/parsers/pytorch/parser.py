from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING, Any

from tempo_dag.ir.graph import Graph
from tempo_dag.parsers.onnx.parser import ONNXParser

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

    from tempo_dag.ir.registry import OperatorRegistry


class PyTorchParser:
    """
    Translates PyTorch modules into the TempoDAG Intermediate Representation (IR).

    It works by first exporting the module to an ONNX model and then parsing
    the resulting ONNX model into an IR Graph.
    """

    def __init__(self, registry: OperatorRegistry | None = None) -> None:
        """Initialize the PyTorch parser with an optional operator registry."""
        self.onnx_parser = ONNXParser(registry=registry)

    def parse_module(
        self,
        module: nn.Module,
        example_input: torch.Tensor | tuple[torch.Tensor, ...],
        **export_kwargs: Any,
    ) -> Graph:
        """
        Export a PyTorch module to an IR Graph.

        Args:
            module: The PyTorch module to export.
            example_input: Example input tensor(s) to trace the model.
            **export_kwargs: Additional arguments to pass to torch.onnx.export.

        Returns:
            A TempoDAG IR Graph representation of the module.
        """
        import torch  # Delayed import for performance if not used

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = os.path.join(tmp_dir, "model.onnx")

            # Default export settings
            default_kwargs = {
                "export_params": True,
                "opset_version": 18,
                "do_constant_folding": True,
                "input_names": ["input"],
                "output_names": ["output"],
            }
            default_kwargs.update(export_kwargs)

            # Normalize example_input to a tuple to satisfy some type-checkers
            args: tuple[Any, ...] = (
                example_input if isinstance(example_input, tuple) else (example_input,)
            )

            torch.onnx.export(module, args, onnx_path, **default_kwargs)

            return self.onnx_parser.parse(onnx_path)
