from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING, Any

from tempo_dag.ir.graph import Graph
from tempo_dag.parsers.onnx.parser import ONNXParser

if TYPE_CHECKING:
    from tempo_dag.ir.registry import OperatorRegistry


class TensorFlowParser:
    """
    Translates TensorFlow/Keras models into the TempoDAG IR.

    It works by first exporting the model to an ONNX model using tf2onnx,
    and then parsing the resulting ONNX model into an IR Graph.
    """

    def __init__(self, registry: OperatorRegistry | None = None) -> None:
        """Initialize the TensorFlow parser with an optional operator registry."""
        self.onnx_parser = ONNXParser(registry=registry)

    def parse_model(
        self,
        model: Any,  # Using Any here as tf might not be installed
        **export_kwargs: Any,
    ) -> Graph:
        """
        Export a TensorFlow/Keras model to an IR Graph.

        Args:
            model: The TensorFlow or Keras model to export.
            **export_kwargs: Additional arguments to pass to tf2onnx.

        Returns:
            A TempoDAG IR Graph representation of the model.
        """
        # Delayed imports as TF is heavy and might be missing in some environments
        try:
            import tensorflow as tf
            from tf2onnx import convert
        except ImportError:
            raise ImportError(
                "TensorFlow and tf2onnx are required for TensorFlowParser. "
                "Install them with `pip install tensorflow tf2onnx`."
            ) from None  # Supress ModuleNotFoundError

        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = os.path.join(tmp_dir, "model.onnx")

            opset = export_kwargs.pop("opset", 14)

            if isinstance(model, tf.keras.Model):
                convert.from_keras(
                    model, opset=opset, output_path=onnx_path, **export_kwargs
                )
            else:
                convert.from_function(
                    model, opset=opset, output_path=onnx_path, **export_kwargs
                )

            if not os.path.exists(onnx_path):
                raise RuntimeError("Failed to export TensorFlow model to ONNX.")

            return self.onnx_parser.parse(onnx_path)
