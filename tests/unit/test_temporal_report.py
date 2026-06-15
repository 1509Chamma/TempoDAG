from typing import cast

from tempo_dag.ir.graph import Graph
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ir_temporal import (
    BufferSpec,
    Edge0,
    EdgeDelta,
    Kernel,
    Process,
    StateKind,
    StateSpec,
    derive_temporal_baseline_report,
)
from tempo_dag.ops.builtins import Add, MatMul


def tensor(value_id: str, shape: list[int]) -> Value:
    return Value(
        value_id=value_id,
        vtype=ValueType.TENSOR,
        dtype="float32",
        shape=shape,
        axes=[f"axis_{idx}" for idx in range(len(shape))],
    )


def test_baseline_report_summarizes_resources_traffic_and_directives() -> None:
    values = {
        "x": tensor("x", [2, 3]),
        "w": tensor("w", [3, 4]),
        "hidden": tensor("hidden", [2, 4]),
        "z": tensor("z", [2, 4]),
        "out": tensor("out", [2, 4]),
    }
    graph = Graph(
        values=values,
        ops={
            "matmul": MatMul("matmul", inputs=["x", "w"], outputs=["z"]),
            "add": Add("add", inputs=["z", "hidden"], outputs=["out"]),
        },
        graph_inputs=["x", "w", "hidden"],
        graph_outputs=["out"],
    )
    process = Process(
        process_id="report_demo",
        kernels={"kernel": Kernel("kernel", graph=graph)},
    )

    report = derive_temporal_baseline_report(
        process,
        trace_metadata={
            "python_latency_ns_per_step": 1200.0,
            "hls_clock_period_ns": 4.0,
        },
    )
    payload = report.to_dict()

    assert payload["process_id"] == "report_demo"
    assert report.summary["estimated_initiation_interval"] == 1
    assert cast(int, report.resource_summary["dsp"]) >= 3
    assert cast(int, report.traffic_summary["total_elements_per_timestep"]) >= 8
    assert report.baseline_comparison["estimated_speedup_vs_python"] is not None
    assert any(row["directive"] == "DATAFLOW" for row in report.directive_plan)
    assert any(row["directive"] == "PIPELINE" for row in report.directive_plan)
    assert report.node_table
    assert report.edge_table


def test_baseline_report_includes_temporal_storage_buffers() -> None:
    graph = Graph(values={}, ops={}, graph_inputs=[], graph_outputs=[])
    process = Process(
        process_id="stateful_report",
        kernels={"kernel": Kernel("kernel", graph=graph)},
        states={
            "hidden": StateSpec(
                state_id="hidden",
                kind=StateKind.HIDDEN,
                dtype="float32",
                shape=(4,),
            )
        },
        buffers={
            "window": BufferSpec(
                buffer_id="window",
                dtype="float32",
                shape=(2,),
                depth=8,
            )
        },
        edge0=[
            Edge0("hidden", "kernel", value_id="hidden_state"),
            Edge0("window", "kernel", value_id="window_view"),
        ],
        edge_delta=[
            EdgeDelta("hidden", "kernel", lag_cycles=1, value_id="h_prev"),
            EdgeDelta("kernel", "hidden", lag_cycles=1, value_id="h_next"),
        ],
    )

    report = derive_temporal_baseline_report(process)

    assert any(row["kind"] == "buffer_read" for row in report.buffer_table)
    assert any(row["kind"] == "state_read" for row in report.buffer_table)
    assert any(row["kind"] == "temporal_delay" for row in report.buffer_table)
    assert any(row["storage_kind"] == "ring_buffer" for row in report.buffer_table)
    assert any(row["storage_kind"] == "register" for row in report.buffer_table)
    assert all(cast(int, row["depth"]) >= 1 for row in report.buffer_table)


def test_baseline_report_disambiguates_reused_value_ids_by_kernel() -> None:
    process = Process(
        process_id="multi_kernel_report",
        kernels={
            "small": Kernel(
                "small",
                graph=Graph(
                    values={
                        "x": tensor("x", [2]),
                        "y": tensor("y", [2]),
                        "out": tensor("out", [2]),
                    },
                    ops={"add": Add("add", inputs=["x", "y"], outputs=["out"])},
                    graph_inputs=["x", "y"],
                    graph_outputs=["out"],
                ),
            ),
            "wide": Kernel(
                "wide",
                graph=Graph(
                    values={
                        "x": tensor("x", [5]),
                        "y": tensor("y", [5]),
                        "out": tensor("out", [5]),
                    },
                    ops={"add": Add("add", inputs=["x", "y"], outputs=["out"])},
                    graph_inputs=["x", "y"],
                    graph_outputs=["out"],
                ),
            ),
        },
    )

    report = derive_temporal_baseline_report(process)
    rows = {row["edge_id"]: row for row in report.edge_table}

    assert rows["small.input->add:x"]["shape"] == [2]
    assert rows["wide.input->add:x"]["shape"] == [5]
