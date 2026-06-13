import onnx

from tempo_dag.parsers.temporal_onnx import (
    TemporalONNXParser,
    build_demo_temporal_onnx_model,
)


def test_temporal_onnx_parser_lowers_demo_model_to_process() -> None:
    parser = TemporalONNXParser()
    model = build_demo_temporal_onnx_model()

    result = parser.parse_model(model, process_id="demo_process")

    assert result.process.process_id == "demo_process"
    assert "kernel_main" in result.process.kernels
    assert "rolling_mean_node_buffer" in result.process.buffers
    assert "rolling_mean_node_rolling_mean" in result.report.states
    assert result.report.buffers == ("rolling_mean_node_buffer",)
    assert result.process.edge_delta[0].lag_cycles == 1


def test_temporal_onnx_parser_detects_scan_state_buffers() -> None:
    helper = onnx.helper
    tensor_proto = onnx.TensorProto
    body = helper.make_graph(
        [helper.make_node("Add", ["state_in", "x_t"], ["state_out"], name="body_add")],
        "scan_body",
        [
            helper.make_tensor_value_info("state_in", tensor_proto.FLOAT, [1]),
            helper.make_tensor_value_info("x_t", tensor_proto.FLOAT, [1]),
        ],
        [helper.make_tensor_value_info("state_out", tensor_proto.FLOAT, [1])],
    )
    scan = helper.make_node(
        "Scan",
        ["state_init", "scan_input"],
        ["state_final", "scan_output"],
        name="scan_node",
        num_scan_inputs=1,
        body=body,
    )
    graph = helper.make_graph(
        [scan],
        "scan_graph",
        [
            helper.make_tensor_value_info("state_init", tensor_proto.FLOAT, [1]),
            helper.make_tensor_value_info("scan_input", tensor_proto.FLOAT, [4, 1]),
        ],
        [helper.make_tensor_value_info("scan_output", tensor_proto.FLOAT, [4, 1])],
    )
    model = helper.make_model(graph, producer_name="scan_temporal")

    result = TemporalONNXParser().parse_model(model, process_id="scan_process")

    assert any(
        buffer_id.startswith("scan_node_") for buffer_id in result.process.buffers
    )
    assert result.report.detected_patterns[0].op_type == "Scan"
