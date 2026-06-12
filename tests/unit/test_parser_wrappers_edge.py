import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from tempo_dag.ir.graph import Graph
from tempo_dag.parsers.pytorch.parser import PyTorchParser
from tempo_dag.parsers.tensorflow.parser import TensorFlowParser

# PyTorch parser tests


class DummyModule(nn.Module):
    def forward(self, x):
        return x


class DummyTupleInputModule(nn.Module):
    def forward(self, x, y):
        return x + y


def test_pytorch_parser_tuple_input():
    parser = PyTorchParser()
    parser.onnx_parser.parse = MagicMock(return_value=Graph({}, {}, [], []))

    # Single input works if model expects one
    model_single = DummyModule()
    parser.parse_module(model_single, torch.randn(1))

    # Tuple input works if it matches model signature
    model_tuple = DummyTupleInputModule()
    parser.parse_module(model_tuple, (torch.randn(1), torch.randn(1)))

    # Check that onnx_parser.parse was called
    assert parser.onnx_parser.parse.call_count == 2


# TensorFlow parser tests


@patch.dict(sys.modules, {"tf2onnx": None, "tensorflow": None})
def test_tensorflow_parser_missing_deps():
    parser = TensorFlowParser()
    with pytest.raises(ImportError, match="TensorFlow and tf2onnx are required"):
        parser.parse_model("dummy_model")


@patch("tf2onnx.convert.from_keras")
@patch("os.path.exists", return_value=False)
def test_tensorflow_parser_export_fails(mock_exists, mock_from_keras):
    # If the file is not created, it should raise RuntimeError
    parser = TensorFlowParser()
    import tensorflow as tf

    model = tf.keras.Sequential()

    with pytest.raises(RuntimeError, match="Failed to export TensorFlow model to ONNX"):
        parser.parse_model(model)


@patch("tf2onnx.convert.from_function")
@patch("os.path.exists", return_value=True)
def test_tensorflow_parser_from_function(mock_exists, mock_from_function):
    parser = TensorFlowParser()
    parser.onnx_parser.parse = MagicMock(return_value=Graph({}, {}, [], []))

    # A generic object that isn't tf.keras.Model should use from_function
    class DummyFuncModel:
        pass

    model = DummyFuncModel()

    parser.parse_model(model)
    mock_from_function.assert_called_once()
    assert parser.onnx_parser.parse.call_count == 1

