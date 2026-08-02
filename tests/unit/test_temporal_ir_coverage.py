"""Coverage for temporal process validation errors, temporal builtin operator
metadata paths, and schedule/report classification branches."""

import pytest

from tempo_dag.ir.graph import Graph
from tempo_dag.ir.op import InvalidOperatorInstanceError
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ir_temporal import (
    BufferSpec,
    Clock,
    Edge0,
    Kernel,
    Process,
    ScheduleEdgeKind,
    StateKind,
    StateSpec,
    TemporalIRValidationError,
    derive_temporal_baseline_report,
    derive_temporal_schedule,
)
from tempo_dag.ops.temporal_builtins import (
    Delay,
    FixedPointRange,
    RollingMean,
    RollingWindow,
    ScanCell,
)


def _empty_graph():
    return Graph(values={}, ops={}, graph_inputs=[], graph_outputs=[])


def _state(state_id, shape=(1,), axes=()):
    return StateSpec(state_id, StateKind.HIDDEN, "float32", shape, axes=axes)


def _buffer(buffer_id, shape=(1,), depth=1, axes=()):
    return BufferSpec(buffer_id, "float32", shape, depth, axes=axes)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: Process(process_id=""), "process_id must be non-empty"),
        (
            lambda: Process(
                process_id="p",
                kernels={"dup": Kernel("dup", _empty_graph())},
                states={"dup": _state("dup")},
            ),
            "globally unique",
        ),
        (
            lambda: Process(process_id="p", states={"s1": _state("s2")}),
            "does not match state_id",
        ),
        (
            lambda: Process(process_id="p", buffers={"b1": _buffer("b2")}),
            "does not match buffer_id",
        ),
        (lambda: Process(process_id="p", clocks={}), "at least one clock"),
        (
            lambda: Process(process_id="p", clocks={"main": Clock("alt")}),
            "does not match clock_id",
        ),
        (
            lambda: Process(process_id="p", clocks={"": Clock("")}),
            "clock_id must be non-empty",
        ),
        (
            lambda: Process(process_id="p", clocks={"main": Clock("main", period=0)}),
            "period must be >= 1",
        ),
        (
            lambda: Process(process_id="p", kernels={"": Kernel("", _empty_graph())}),
            "kernel_id must be non-empty",
        ),
        (
            lambda: Process(process_id="p", states={"": _state("")}),
            "state_id must be non-empty",
        ),
        (
            lambda: Process(process_id="p", states={"s": _state("s", shape=(0,))}),
            "shape dimensions must be >= 1",
        ),
        (
            lambda: Process(process_id="p", buffers={"": _buffer("")}),
            "buffer_id must be non-empty",
        ),
        (
            lambda: Process(process_id="p", buffers={"b": _buffer("b", depth=0)}),
            "depth must be >= 1",
        ),
        (
            lambda: Process(
                process_id="p",
                buffers={"b": _buffer("b", shape=(1, 2), axes=("x",))},
            ),
            "shape and axes lengths differ",
        ),
        (
            lambda: Process(process_id="p", buffers={"b": _buffer("b", shape=(0,))}),
            "shape dimensions must be >= 1",
        ),
        (
            lambda: Process(process_id="p", edge0=[Edge0("a", "b")]),
            "edges require at least one component",
        ),
    ],
)
def test_temporal_process_validation_errors(factory, match):
    with pytest.raises(TemporalIRValidationError, match=match):
        factory().validate()


# ---------------------------------------------------------------------------
# temporal builtin operators
# ---------------------------------------------------------------------------


def _tensor(value_id, shape=(2,)):
    return Value(
        value_id=value_id,
        vtype=ValueType.TENSOR,
        dtype="float32",
        shape=list(shape),
        axes=[f"a{i}" for i in range(len(shape))],
    )


@pytest.fixture
def unary_values():
    return {"x": _tensor("x"), "y": _tensor("y")}


def test_fixed_point_range_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="minimum must be <= maximum"):
        FixedPointRange(minimum=2.0, maximum=1.0)


def test_fixed_point_range_to_dict():
    assert FixedPointRange(minimum=-1.0, maximum=1.0).to_dict() == {
        "minimum": -1.0,
        "maximum": 1.0,
        "signed": True,
    }


def test_rolling_window_rejects_non_positive_window_size():
    operator = RollingWindow("rw", ["x"], ["y"], attrs={"window_size": 0})
    with pytest.raises(InvalidOperatorInstanceError, match="window_size >= 1"):
        operator.validate({})


def test_rolling_stat_rejects_non_positive_window_size(unary_values):
    operator = RollingMean("rm", ["x"], ["y"], attrs={"window_size": 0})
    with pytest.raises(InvalidOperatorInstanceError, match="window_size >= 1"):
        operator.validate(unary_values)


def test_scan_cell_estimates_cost_from_output_work(unary_values):
    cost = ScanCell("sc", ["x"], ["y"]).estimate_fpga_cost(unary_values)
    assert cost.latency_cycles == 2
    assert cost.lut == 2
    assert cost.ff == 2
    assert cost.metadata == {"heuristic": "scan_cell"}


def test_fixed_point_ranges_attr_must_be_mapping(unary_values):
    operator = ScanCell("sc", ["x"], ["y"], attrs={"fixed_point_ranges": "oops"})
    with pytest.raises(InvalidOperatorInstanceError, match="must be a mapping"):
        operator.temporal_metadata(unary_values)


def test_delay_rejects_blank_buffer_id_attr(unary_values):
    operator = Delay("dl", ["x"], ["y"], attrs={"buffer_id": "   "})
    with pytest.raises(InvalidOperatorInstanceError, match="buffer_id must be"):
        operator.temporal_metadata(unary_values)


def test_scan_cell_rejects_non_sequence_state_ids(unary_values):
    operator = ScanCell("sc", ["x"], ["y"], attrs={"state_ids": "h"})
    with pytest.raises(InvalidOperatorInstanceError, match="sequence of strings"):
        operator.temporal_metadata(unary_values)


def test_scan_cell_rejects_blank_state_id_entry(unary_values):
    operator = ScanCell("sc", ["x"], ["y"], attrs={"state_ids": ["ok", ""]})
    with pytest.raises(
        InvalidOperatorInstanceError, match=r"state_ids\[1\] must be a non-empty"
    ):
        operator.temporal_metadata(unary_values)


def test_scan_cell_with_state_ids_reports_stateful_metadata(unary_values):
    operator = ScanCell("sc", ["x"], ["y"], attrs={"state_ids": ["h0", "h1"]})
    metadata = operator.temporal_metadata(unary_values)
    assert metadata.stateful is True
    assert metadata.state_reads == ("h0", "h1")
    assert metadata.state_writes == ("h0", "h1")
    assert metadata.lag_cycles == 1


# ---------------------------------------------------------------------------
# schedule edge classification + report value metadata fallback
# ---------------------------------------------------------------------------


@pytest.fixture
def write_edge_process():
    process = Process(
        process_id="edge_cov",
        kernels={
            "k1": Kernel("k1", _empty_graph()),
            "k2": Kernel("k2", _empty_graph()),
        },
        states={"s": _state("s")},
        buffers={"buf": _buffer("buf", depth=2)},
        edge0=[Edge0("k1", "s"), Edge0("k1", "buf"), Edge0("k1", "k2")],
    )
    process.validate()
    return process


def test_schedule_classifies_state_and_buffer_writes(write_edge_process):
    schedule = derive_temporal_schedule(write_edge_process)
    kinds = {edge.edge_id: edge for edge in schedule.edges}

    state_write = kinds["k1->s:value"]
    assert state_write.kind is ScheduleEdgeKind.STATE_WRITE
    assert state_write.storage_kind is not None

    buffer_write = kinds["k1->buf:value"]
    assert buffer_write.kind is ScheduleEdgeKind.BUFFER_WRITE
    assert buffer_write.storage_kind is not None


def test_report_falls_back_to_unknown_value_metadata(write_edge_process):
    schedule = derive_temporal_schedule(write_edge_process)
    report = derive_temporal_baseline_report(write_edge_process, schedule)
    kernel_to_kernel = next(
        row for row in report.edge_table if row["edge_id"] == "k1->k2:value"
    )
    assert kernel_to_kernel["dtype"] == "unknown"
    assert kernel_to_kernel["shape"] == []
    assert kernel_to_kernel["elements_per_timestep"] == 1
