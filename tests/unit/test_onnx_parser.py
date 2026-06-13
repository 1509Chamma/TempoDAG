import unittest

import onnx.helper as helper
from onnx import TensorProto

from tempo_dag.parsers.onnx.parser import ONNXParser


class TestONNXParser(unittest.TestCase):
    def setUp(self):
        self.parser = ONNXParser()

    def test_matmul_add_parse(self):
        # Create a simple ONNX model: MatMul + Add
        node1 = helper.make_node(
            "MatMul", ["X", "W"], ["MatMul_out"], name="matmul_node"
        )
        node2 = helper.make_node("Add", ["MatMul_out", "B"], ["Y"], name="add_node")

        graph = helper.make_graph(
            [node1, node2],
            "test_graph",
            [
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 10]),
                helper.make_tensor_value_info("W", TensorProto.FLOAT, [10, 20]),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 20]),
            ],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 20])],
        )
        model = helper.make_model(graph, producer_name="test")

        ir_graph = self.parser.parse_model(model)

        self.assertEqual(len(ir_graph.ops), 2)
        self.assertIn("matmul_node", ir_graph.ops)
        self.assertIn("add_node", ir_graph.ops)

        self.assertEqual(ir_graph.ops["matmul_node"].op_type, "MatMul")
        self.assertEqual(ir_graph.ops["add_node"].op_type, "Add")

        self.assertIn("X", ir_graph.values)
        self.assertEqual(ir_graph.values["X"].shape, [1, 10])

    def test_lstm_parse(self):
        # Create a simple ONNX LSTM model
        hidden_size = 32
        input_size = 16
        node = helper.make_node(
            "LSTM", ["X", "W", "R"], ["Y"], name="lstm_node", hidden_size=hidden_size
        )

        graph = helper.make_graph(
            [node],
            "lstm_graph",
            [
                helper.make_tensor_value_info(
                    "X", TensorProto.FLOAT, [5, 1, input_size]
                ),
                helper.make_tensor_value_info(
                    "W", TensorProto.FLOAT, [1, 4 * hidden_size, input_size]
                ),
                helper.make_tensor_value_info(
                    "R", TensorProto.FLOAT, [1, 4 * hidden_size, hidden_size]
                ),
            ],
            [
                helper.make_tensor_value_info(
                    "Y", TensorProto.FLOAT, [5, 1, 1, hidden_size]
                )
            ],
        )
        model = helper.make_model(graph, producer_name="lstm_test")

        ir_graph = self.parser.parse_model(model)

        self.assertIn("lstm_node", ir_graph.ops)
        lstm_op = ir_graph.ops["lstm_node"]
        self.assertEqual(lstm_op.op_type, "LSTM")
        self.assertEqual(lstm_op.attrs["hidden_size"], hidden_size)

    def test_sigmoid_parse(self):
        # Create a simple ONNX Sigmoid model
        node = helper.make_node("Sigmoid", ["X"], ["Y"], name="sigmoid_node")

        graph = helper.make_graph(
            [node],
            "sigmoid_graph",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [10, 20])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [10, 20])],
        )
        model = helper.make_model(graph, producer_name="sigmoid_test")

        ir_graph = self.parser.parse_model(model)

        self.assertIn("sigmoid_node", ir_graph.ops)
        self.assertEqual(ir_graph.ops["sigmoid_node"].op_type, "Sigmoid")

    def test_detects_scan_temporal_pattern(self):
        body_node = helper.make_node("Add", ["state_in", "x_t"], ["state_out"])
        body = helper.make_graph(
            [body_node],
            "scan_body",
            [
                helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [4]),
                helper.make_tensor_value_info("x_t", TensorProto.FLOAT, [4]),
            ],
            [helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [4])],
        )
        scan = helper.make_node(
            "Scan",
            ["initial_state", "sequence"],
            ["final_state", "scan_outputs"],
            name="scan_node",
            num_scan_inputs=1,
            body=body,
        )
        graph = helper.make_graph(
            [scan],
            "scan_graph",
            [
                helper.make_tensor_value_info("initial_state", TensorProto.FLOAT, [4]),
                helper.make_tensor_value_info("sequence", TensorProto.FLOAT, [8, 4]),
            ],
            [
                helper.make_tensor_value_info("final_state", TensorProto.FLOAT, [4]),
                helper.make_tensor_value_info(
                    "scan_outputs", TensorProto.FLOAT, [8, 4]
                ),
            ],
        )
        model = helper.make_model(graph, producer_name="scan_test")

        patterns = self.parser.detect_temporal_patterns(model)

        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].op_type, "Scan")
        self.assertEqual(patterns[0].stateful_inputs, ("initial_state",))
        self.assertEqual(patterns[0].body_node_count, 1)

    def test_detects_loop_temporal_pattern(self):
        body_node = helper.make_node("Add", ["state_in", "state_in"], ["state_out"])
        body = helper.make_graph(
            [body_node],
            "loop_body",
            [
                helper.make_tensor_value_info("iter", TensorProto.INT64, []),
                helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
                helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [4]),
            ],
            [
                helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
                helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [4]),
            ],
        )
        loop = helper.make_node(
            "Loop",
            ["trip_count", "cond", "loop_state"],
            ["final_state"],
            name="loop_node",
            body=body,
        )
        graph = helper.make_graph(
            [loop],
            "loop_graph",
            [
                helper.make_tensor_value_info("trip_count", TensorProto.INT64, []),
                helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
                helper.make_tensor_value_info("loop_state", TensorProto.FLOAT, [4]),
            ],
            [helper.make_tensor_value_info("final_state", TensorProto.FLOAT, [4])],
        )
        model = helper.make_model(graph, producer_name="loop_test")

        patterns = self.parser.detect_temporal_patterns(model)

        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].op_type, "Loop")
        self.assertEqual(patterns[0].stateful_inputs, ("loop_state",))
        self.assertEqual(patterns[0].body_node_count, 1)


if __name__ == "__main__":
    unittest.main()
