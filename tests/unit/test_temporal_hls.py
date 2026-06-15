import json
import tempfile
from dataclasses import replace
from pathlib import Path

from tempo_dag.codegen.hls.temporal_generator import (
    TemporalArtifactKind,
    render_temporal_artifact_from_trace,
    render_temporal_process_hls,
    write_temporal_hls_artifact_bundle,
)
from tempo_dag.examples import rolling_window_process
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
    assert "Running temporal golden trace (" in artifact.testbench_hls


def test_render_temporal_process_hls_includes_half_header_for_float16_buffers() -> None:
    process = (
        TemporalONNXParser()
        .parse_model(
            build_demo_temporal_onnx_model(),
            process_id="demo_process",
        )
        .process
    )
    buffer_id = next(iter(process.buffers))
    process.buffers[buffer_id] = replace(process.buffers[buffer_id], dtype="float16")

    rendered = render_temporal_process_hls(process)

    assert "#include <hls_half.h>" in rendered
    assert "static half" in rendered


def test_render_temporal_process_hls_includes_execution_contract_comments() -> None:
    rendered = render_temporal_process_hls(rolling_window_process())

    assert "// reset_policy: zero" in rendered
    assert "// warmup_timesteps: 7" in rendered
    assert "// flush_cycles: 0" in rendered
    assert "// schedule_estimated_latency_cycles:" in rendered
    assert "// schedule_estimated_initiation_interval: 1" in rendered
    assert "// edge_delta_storage: feature_kernel->rolling_window@1:x" in rendered
    assert "-> register latency=1" in rendered
    assert "// buffer_storage: rolling_window -> ring_buffer latency=8" in rendered
    assert "// schedule_node: feature_kernel kind=kernel phase=1" in rendered
    assert "// schedule_node: rolling_window kind=buffer phase=0" in rendered
    assert (
        "// schedule_edge: feature_kernel->rolling_window@1:x " "kind=temporal_delay"
    ) in rendered
    assert "#pragma HLS DATAFLOW" in rendered


def test_write_temporal_hls_artifact_bundle_emits_manifest_and_files() -> None:
    process = (
        TemporalONNXParser()
        .parse_model(
            build_demo_temporal_onnx_model(),
            process_id="demo_process",
        )
        .process
    )
    trace = load_golden_trace("tests/verification/golden_traces/rolling_mean.json")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        manifest = write_temporal_hls_artifact_bundle(
            process,
            trace,
            output_dir,
            stem="demo",
        )

        files = {artifact.kind: artifact.path for artifact in manifest.files}
        assert set(files) == {
            TemporalArtifactKind.PROCESS_JSON,
            TemporalArtifactKind.GOLDEN_TRACE_JSON,
            TemporalArtifactKind.SCHEDULE_JSON,
            TemporalArtifactKind.BASELINE_REPORT_JSON,
            TemporalArtifactKind.PROCESS_HLS,
            TemporalArtifactKind.TESTBENCH_HLS,
            TemporalArtifactKind.MANIFEST_JSON,
        }
        for path in files.values():
            assert path.is_file()

        manifest_payload = json.loads(
            files[TemporalArtifactKind.MANIFEST_JSON].read_text()
        )
        schedule_payload = json.loads(
            files[TemporalArtifactKind.SCHEDULE_JSON].read_text()
        )
        report_payload = json.loads(
            files[TemporalArtifactKind.BASELINE_REPORT_JSON].read_text()
        )
        assert manifest_payload["process_id"] == "demo_process"
        assert schedule_payload["process_id"] == "demo_process"
        assert report_payload["process_id"] == "demo_process"
        assert schedule_payload["nodes"]
        assert schedule_payload["edges"]
        assert report_payload["node_table"]
        assert report_payload["edge_table"]
        assert report_payload["directive_plan"]
        assert {item["kind"] for item in manifest_payload["files"]} == {
            kind.value for kind in TemporalArtifactKind
        }
