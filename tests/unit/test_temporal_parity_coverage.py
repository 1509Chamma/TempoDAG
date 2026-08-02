"""Coverage-focused tests for tempo_dag.verification.temporal_parity."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tempo_dag.quantization_config import (
    FixedPointSpec,
    OverflowPolicy,
    QuantizationScheme,
    QuantizationSpec,
    QuantizationType,
    StateQuantSpec,
)
from tempo_dag.verification import temporal_parity
from tempo_dag.verification.temporal_parity import (
    FixedPointOracle,
    StreamingPyTorchAdapter,
    TemporalExecutionTrace,
    TemporalTraceStep,
)

OUTPUT_SPEC = QuantizationSpec(
    bit_width=8,
    scheme=QuantizationScheme.SYMMETRIC,
    qtype=QuantizationType.FIXED_POINT,
    fixed_point=FixedPointSpec(integer_bits=4, fractional_bits=4),
    scale=2**-4,
    zero_point=0,
)


def _narrow_state_spec(policy: OverflowPolicy) -> StateQuantSpec:
    return StateQuantSpec(
        dtype="fixed4",
        scale=0.25,
        overflow_policy=policy,
        fixed_point=FixedPointSpec(integer_bits=2, fractional_bits=2),
    )


def _single_step_trace(outputs=None, state=None):
    step = TemporalTraceStep(
        timestep=0,
        inputs={},
        outputs={name: np.asarray(value) for name, value in (outputs or {}).items()},
        state={name: np.asarray(value) for name, value in (state or {}).items()},
    )
    return TemporalExecutionTrace((step,))


def test_execution_trace_to_dict_serializes_steps() -> None:
    step = TemporalTraceStep(
        timestep=0,
        inputs={"x": np.array([1.0])},
        outputs={"y": np.array([2.0])},
        state={"s": np.array([3.0])},
    )

    payload = TemporalExecutionTrace((step,)).to_dict()

    assert payload == {
        "steps": [
            {
                "timestep": 0,
                "inputs": {"x": [1.0]},
                "outputs": {"y": [2.0]},
                "state": {"s": [3.0]},
            }
        ]
    }


def test_oracle_copies_state_without_matching_spec() -> None:
    oracle = FixedPointOracle()
    trace = _single_step_trace(outputs={"y": [0.3]}, state={"s": [0.7]})

    quantized = oracle.quantize_trace(trace)

    result_state = quantized.steps[0].state["s"]
    np.testing.assert_array_equal(result_state, np.array([0.7]))
    assert result_state is not trace.steps[0].state["s"]


def test_oracle_infers_fixed_point_spec_from_scale() -> None:
    spec = StateQuantSpec(dtype="fixed", scale=2**-4)
    oracle = FixedPointOracle(state_specs={"s": spec})
    trace = _single_step_trace(state={"s": [0.3]})

    quantized = oracle.quantize_trace(trace)

    # 0.3 / 2**-4 = 4.8 rounds to 5 -> dequantizes to 0.3125.
    np.testing.assert_allclose(quantized.steps[0].state["s"], np.array([0.3125]))


def test_adapter_rejects_mapping_result_without_output_payload() -> None:
    class NoOutputModel:
        def __call__(self, item):
            return {"state": {"s": np.array([1.0])}}

    adapter = StreamingPyTorchAdapter(NoOutputModel())

    with pytest.raises(ValueError, match="output payload"):
        adapter.run_sequence([np.array([1.0])])


def test_adapter_wraps_raw_result_as_single_output() -> None:
    class RawArrayModel:
        def __call__(self, item):
            return np.array([3.0])

    trace = StreamingPyTorchAdapter(RawArrayModel()).run_sequence([np.array([1.0])])

    assert trace.steps[0].outputs["output"].tolist() == [3.0]
    assert trace.steps[0].state == {}


def test_adapter_rejects_non_mapping_outputs_payload() -> None:
    class BadOutputsModel:
        def __call__(self, item):
            return {"outputs": [1.0, 2.0]}

    adapter = StreamingPyTorchAdapter(BadOutputsModel())

    with pytest.raises(ValueError, match="mapping of named arrays"):
        adapter.run_sequence([np.array([1.0])])


def test_adapter_treats_none_tuple_state_as_empty() -> None:
    class TupleNoneStateModel:
        def __call__(self, item):
            return (np.array([2.0]), None)

    trace = StreamingPyTorchAdapter(TupleNoneStateModel()).run_sequence(
        [np.array([1.0])]
    )

    assert trace.steps[0].outputs["output"].tolist() == [2.0]
    assert trace.steps[0].state == {}


def test_adapter_passes_non_array_inputs_through_unchanged() -> None:
    class DoublingModel:
        def __init__(self) -> None:
            self.seen: list[object] = []

        def __call__(self, item):
            self.seen.append(item)
            return item * 2.0

    model = DoublingModel()
    trace = StreamingPyTorchAdapter(model).run_sequence([2.0, 3.0])

    assert model.seen == [2.0, 3.0]
    assert all(isinstance(item, float) for item in model.seen)
    assert trace.steps[1].outputs["output"].tolist() == 6.0


def test_adapter_calls_reset_and_eval_and_reads_named_output_mapping() -> None:
    class StatefulModel:
        def __init__(self) -> None:
            self.reset_calls = 0
            self.eval_calls = 0

        def reset_state(self) -> None:
            self.reset_calls += 1

        def eval(self) -> None:
            self.eval_calls += 1

        def __call__(self, item):
            return {"output": np.array([1.0]), "state": {"h": np.array([0.5])}}

    model = StatefulModel()
    trace = StreamingPyTorchAdapter(model).run_sequence([np.array([1.0])])

    assert model.reset_calls == 1
    assert model.eval_calls == 1
    assert trace.steps[0].outputs["output"].tolist() == [1.0]
    assert trace.steps[0].state["h"].tolist() == [0.5]


def test_adapter_normalizes_named_outputs_mapping() -> None:
    class NamedOutputsModel:
        def __call__(self, item):
            return {
                "outputs": {"y": np.array([2.0])},
                "state": {"s": np.array([1.0])},
            }

    trace = StreamingPyTorchAdapter(NamedOutputsModel()).run_sequence([np.array([1.0])])

    assert trace.steps[0].outputs["y"].tolist() == [2.0]
    assert trace.steps[0].state["s"].tolist() == [1.0]


def test_adapter_wraps_bare_tuple_state_under_state_name() -> None:
    class TupleStateModel:
        def __call__(self, item):
            return (np.array([2.0]), np.array([7.0]))

    trace = StreamingPyTorchAdapter(TupleStateModel()).run_sequence([np.array([1.0])])

    assert trace.steps[0].state["state"].tolist() == [7.0]


def test_oracle_quantizes_outputs_with_matching_spec() -> None:
    oracle = FixedPointOracle(output_specs={"y": OUTPUT_SPEC})
    trace = _single_step_trace(outputs={"y": [0.3]})

    quantized = oracle.quantize_trace(trace)

    # 0.3 * 16 = 4.8 rounds to 5 -> dequantizes to 0.3125.
    np.testing.assert_allclose(quantized.steps[0].outputs["y"], np.array([0.3125]))


def test_oracle_error_policy_raises_on_state_overflow() -> None:
    oracle = FixedPointOracle(
        state_specs={"s": _narrow_state_spec(OverflowPolicy.ERROR)}
    )
    trace = _single_step_trace(state={"s": [10.0]})

    with pytest.raises(ValueError, match="overflowed fixed-point range"):
        oracle.quantize_trace(trace)


def test_oracle_wrap_policy_wraps_overflowed_state() -> None:
    oracle = FixedPointOracle(
        state_specs={"s": _narrow_state_spec(OverflowPolicy.WRAP)}
    )
    trace = _single_step_trace(state={"s": [2.5]})

    quantized = oracle.quantize_trace(trace)

    # 2.5 / 0.25 = 10 wraps modulo 16 into [-8, 7]: ((10 + 8) % 16) - 8 = -6,
    # which dequantizes to -6 * 0.25 = -1.5.
    np.testing.assert_allclose(quantized.steps[0].state["s"], np.array([-1.5]))


def test_wrap_fixed_point_copies_value_when_fixed_point_missing(monkeypatch) -> None:
    # Defensive branch: _state_spec_to_quant_spec always resolves a fixed-point
    # spec in practice, so force the None case to pin the fallback behavior.
    spec = StateQuantSpec(dtype="fixed", scale=0.5)
    monkeypatch.setattr(
        temporal_parity,
        "_state_spec_to_quant_spec",
        lambda _spec: SimpleNamespace(fixed_point=None),
    )
    value = np.array([1.5, -2.0])

    result = temporal_parity._wrap_fixed_point(value, spec)

    np.testing.assert_array_equal(result, value)
    assert result is not value
