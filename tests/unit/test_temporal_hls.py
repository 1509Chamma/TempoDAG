from tempo_dag.codegen.hls.temporal_generator import (
    render_temporal_artifact_from_trace,
    render_temporal_process_hls,
)
from tempo_dag.parsers.temporal_onnx import (
    TemporalONNXParser,
    build_demo_temporal_onnx_model,
)
from tempo_dag.verification.golden_trace import load_golden_trace


def test_render_temporal_process_hls_includes_buffers_and_step_function() -> None:
    process = (
        TemporalONNXParser()
        .parse_model(
            build_demo_temporal_onnx_model(),
            process_id="demo_process",
        )
        .process
    )

    rendered = render_temporal_process_hls(process)

    assert "Temporal process: demo_process" in rendered
    assert "static float rolling_mean_node_buffer" in rendered
    assert "void demo_process_step()" in rendered


def test_render_temporal_artifact_includes_testbench_trace_comments() -> None:
    process = (
        TemporalONNXParser()
        .parse_model(
            build_demo_temporal_onnx_model(),
            process_id="demo_process",
        )
        .process
    )
    trace = load_golden_trace("tests/verification/golden_traces/rolling_mean.json")

    artifact = render_temporal_artifact_from_trace(process, trace)

    assert "expected output output" in artifact.testbench_hls
    assert "demo_process_step();" in artifact.testbench_hls
    assert "RollingMean attrs" in artifact.process_hls
