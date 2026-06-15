import json
import tempfile
from pathlib import Path

from tempo_dag.codegen.hls.temporal_generator import load_and_render_temporal_artifact
from tempo_dag.examples.temporal_demo import run_demo
from tempo_dag.parsers.temporal_onnx import (
    TemporalONNXParser,
    build_demo_temporal_onnx_model,
)


def test_temporal_demo_pipeline_emits_artifacts() -> None:
    output_dir = _make_output_dir("pipeline")
    report = run_demo(output_dir)

    assert report.validation_passed is True
    assert report.num_trace_steps == 4
    assert (output_dir / "temporal_demo.cpp").is_file()
    assert (output_dir / "temporal_demo_tb.cpp").is_file()
    assert (output_dir / "temporal_demo_trace.json").is_file()
    assert (output_dir / "temporal_demo_process.json").is_file()
    assert (output_dir / "temporal_demo_schedule.json").is_file()
    assert (output_dir / "temporal_demo_baseline_report.json").is_file()
    assert (output_dir / "temporal_demo_report.json").is_file()
    assert (output_dir / "temporal_demo_manifest.json").is_file()
    assert report.manifest_path == str(output_dir / "temporal_demo_manifest.json")

    baseline_report = json.loads(
        (output_dir / "temporal_demo_baseline_report.json").read_text()
    )
    assert baseline_report["baseline_comparison"]["python_latency_ns_per_step"] > 0
    assert baseline_report["baseline_comparison"]["estimated_speedup_vs_python"] > 0


def test_temporal_lowering_connects_to_trace_driven_hls() -> None:
    output_dir = _make_output_dir("hls")
    process = (
        TemporalONNXParser()
        .parse_model(
            build_demo_temporal_onnx_model(),
            process_id="integration_demo",
        )
        .process
    )
    report = run_demo(output_dir)

    artifact = load_and_render_temporal_artifact(
        process,
        output_dir / "temporal_demo_trace.json",
    )

    assert report.validation_passed is True
    assert "void integration_demo_step()" in artifact.process_hls
    assert "expected output output" in artifact.testbench_hls


def _make_output_dir(name: str) -> Path:
    base_dir = Path(".pytest_cache") / "temporal_demo"
    base_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{name}-", dir=base_dir))
