from unittest.mock import MagicMock, patch

import tensorflow as tf

from tempo_dag.ir.graph import Graph
from tempo_dag.parsers.tensorflow.parser import TensorFlowParser


def test_tensorflow_parser_init():
    parser = TensorFlowParser()
    assert parser.onnx_parser is not None


@patch("tf2onnx.convert.from_keras")
@patch("os.path.exists")
def test_parse_model_calls_tf2onnx(mock_exists, mock_tf2onnx):
    # Setup mocks
    mock_exists.return_value = True
    # Create real but simple model
    model = tf.keras.Sequential([tf.keras.layers.Dense(2, input_shape=(3,))])

    parser = TensorFlowParser()
    # Mocked onnx parser to avoid calling real ONNX logic
    parser.onnx_parser.parse = MagicMock(
        return_value=Graph(values={}, ops={}, graph_inputs=[], graph_outputs=[])
    )

    # Act
    graph = parser.parse_model(model)

    # Assert
    assert isinstance(graph, Graph)
    mock_tf2onnx.assert_called_once()
