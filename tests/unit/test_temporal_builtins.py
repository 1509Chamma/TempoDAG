import pytest

from tempo_dag.ir.op import FPGACost, InvalidOperatorInstanceError
from tempo_dag.ir.registry import OperatorRegistry
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ops.temporal_builtins import (
    Delay,
    RollingMean,
    RollingVar,
    RollingWindow,
    TemporalAdd,
    TemporalMatMul,
    register_temporal_builtin_operators,
)


def make_tensor(
    value_id: str,
    shape: list[int],
    axes: list[str] | None = None,
    dtype: str = "float32",
) -> Value:
    if axes is None:
        axes = [f"axis_{idx}" for idx in range(len(shape))]
    return Value(
        value_id=value_id,
        vtype=ValueType.TENSOR,
        dtype=dtype,
        shape=shape,
        axes=axes,
    )


def test_temporal_add_and_matmul_report_stateless_metadata() -> None:
    add_values = {
        "lhs": make_tensor("lhs", [2, 3], ["batch", "feature"]),
        "rhs": make_tensor("rhs", [2, 3], ["batch", "feature"]),
        "out": make_tensor("out", [2, 3], ["batch", "feature"]),
    }
    add = TemporalAdd(
        op_id="add_0",
        inputs=["lhs", "rhs"],
        outputs=["out"],
        attrs={"fixed_point_ranges": {"out": {"minimum": -2.0, "maximum": 2.0}}},
    )

    metadata = add.temporal_metadata(add_values)

    assert metadata.stateful is False
    assert metadata.state_reads == ()
    assert metadata.fixed_point_ranges["out"].minimum == -2.0

    matmul_values = {
        "lhs": make_tensor("lhs", [2, 3], ["rows", "inner"]),
        "rhs": make_tensor("rhs", [3, 4], ["inner", "cols"]),
        "out": make_tensor("out", [2, 4], ["rows", "cols"]),
    }
    matmul = TemporalMatMul(op_id="matmul_0", inputs=["lhs", "rhs"], outputs=["out"])

    assert matmul.temporal_metadata(matmul_values).to_dict()["stateful"] is False


def test_delay_requires_positive_lag_and_threads_buffer_state() -> None:
    values = {
        "x": make_tensor("x", [4], ["feature"]),
        "y": make_tensor("y", [4], ["feature"]),
    }
    delay = Delay(
        op_id="delay_0",
        inputs=["x"],
        outputs=["y"],
        attrs={"lag_cycles": 3, "buffer_id": "x_delay"},
    )

    delay.validate(values)
    metadata = delay.temporal_metadata(values)

    assert metadata.stateful is True
    assert metadata.state_reads == ("x_delay",)
    assert metadata.state_writes == ("x_delay",)
    assert metadata.lag_cycles == 3
    assert delay.estimate_fpga_cost(values) == FPGACost(
        latency_cycles=1,
        initiation_interval=1,
        bram=1,
        lut=2,
        ff=4,
        metadata={"heuristic": "temporal_delay", "lag_cycles": 3},
    )

    bad_delay = Delay(
        op_id="bad_delay",
        inputs=["x"],
        outputs=["y"],
        attrs={"lag_cycles": 0},
    )
    with pytest.raises(InvalidOperatorInstanceError, match="lag_cycles >= 1"):
        bad_delay.validate(values)


def test_rolling_window_adds_causal_window_axis() -> None:
    values = {
        "x": make_tensor("x", [2, 3], ["batch", "feature"]),
        "window": make_tensor("window", [4, 2, 3], ["window", "batch", "feature"]),
    }
    op = RollingWindow(
        op_id="window_0",
        inputs=["x"],
        outputs=["window"],
        attrs={"window_size": 4},
    )

    op.validate(values)
    metadata = op.temporal_metadata(values)

    assert metadata.window_size == 4
    assert metadata.buffers == ("window_0_buffer",)
    assert op.estimate_fpga_cost(values).metadata == {
        "heuristic": "rolling_window",
        "window_size": 4,
    }


def test_rolling_stats_keep_input_shape_and_report_state_ids() -> None:
    values = {
        "x": make_tensor("x", [2, 3], ["batch", "feature"]),
        "mean": make_tensor("mean", [2, 3], ["batch", "feature"]),
        "var": make_tensor("var", [2, 3], ["batch", "feature"]),
    }
    mean = RollingMean(
        op_id="mean_0",
        inputs=["x"],
        outputs=["mean"],
        attrs={"window_size": 8, "state_id": "running_sum"},
    )
    var = RollingVar(
        op_id="var_0",
        inputs=["x"],
        outputs=["var"],
        attrs={"window_size": 8, "state_id": "running_var"},
    )

    mean.validate(values)
    var.validate(values)

    assert mean.temporal_metadata(values).state_reads == (
        "running_sum",
        "mean_0_buffer",
    )
    assert var.temporal_metadata(values).state_writes == (
        "running_var",
        "var_0_buffer",
    )
    assert var.estimate_fpga_cost(values).metadata == {
        "heuristic": "rolling_var",
        "window_size": 8,
    }


def test_temporal_builtins_register_with_operator_registry() -> None:
    registry = OperatorRegistry()

    register_temporal_builtin_operators(registry)

    operator = registry.create(
        "Delay",
        op_id="delay_0",
        inputs=["x"],
        outputs=["y"],
        attrs={"lag_cycles": 2},
    )
    assert isinstance(operator, Delay)
