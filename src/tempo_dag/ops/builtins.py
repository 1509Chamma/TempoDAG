from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, TypeGuard

from tempo_dag.ir.op import FPGACost, InvalidOperatorInstanceError, Operator
from tempo_dag.ir.value import Value, ValueType

if TYPE_CHECKING:
    from tempo_dag.ir.registry import OperatorRegistry


def _snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _shape_product(shape: Sequence[int]) -> int:
    if not shape:
        return 1

    product = 1
    for dim in shape:
        if dim <= 0:
            raise InvalidOperatorInstanceError(
                f"shape dimensions must be positive integers, got {list(shape)}"
            )
        product *= dim
    return product


def _cpp_dtype(dtype: str) -> str:
    return {
        "float16": "half",
        "float32": "float",
        "float64": "double",
        "int16": "std::int16_t",
        "int32": "std::int32_t",
        "int64": "std::int64_t",
    }.get(dtype, dtype)


def _cpp_bool(value: bool) -> str:
    return "true" if value else "false"


def _is_int_sequence(value: object) -> TypeGuard[Sequence[int]]:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and all(isinstance(item, int) for item in value)
    )


class BuiltinOperator(Operator):
    """Shared helpers for built-in primitive operators."""

    HLS_TEMPLATE_DIR: ClassVar[str] = "hls/operators"

    def hls_template_path(self) -> str:
        return f"{self.HLS_TEMPLATE_DIR}/{_snake_case(self.op_type)}.cpp.tpl"

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        input_values = [
            self._lookup_value(values, value_id, "input") for value_id in self.inputs
        ]
        output_values = [
            self._lookup_value(values, value_id, "output") for value_id in self.outputs
        ]
        primary_output = output_values[0] if output_values else None
        return {
            "op_id": self.op_id,
            "op_type": self.op_type,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attrs": dict(self.attrs),
            "input_shapes": [value.shape for value in input_values],
            "output_shapes": [value.shape for value in output_values],
            "cpp_dtype": _cpp_dtype(
                primary_output.dtype
                if primary_output is not None
                else (input_values[0].dtype if input_values else "float32")
            ),
            "input_0_size": (
                _shape_product(input_values[0].shape) if len(input_values) > 0 else 0
            ),
            "input_1_size": (
                _shape_product(input_values[1].shape) if len(input_values) > 1 else 0
            ),
            "output_0_size": (
                _shape_product(output_values[0].shape) if len(output_values) > 0 else 0
            ),
            "has_scalar_lhs": _cpp_bool(
                len(input_values) > 0 and input_values[0].vtype is ValueType.SCALAR
            ),
            "has_scalar_rhs": _cpp_bool(
                len(input_values) > 1 and input_values[1].vtype is ValueType.SCALAR
            ),
        }

    def _require_input_count(self, allowed: int | Iterable[int]) -> None:
        expected = {allowed} if isinstance(allowed, int) else set(allowed)
        if len(self.inputs) not in expected:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} expects {self._format_expected(expected)} inputs, "
                f"got {len(self.inputs)}"
            )

    def _require_output_count(self, allowed: int | Iterable[int]) -> None:
        expected = {allowed} if isinstance(allowed, int) else set(allowed)
        if len(self.outputs) not in expected:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} expects {self._format_expected(expected)} outputs, "
                f"got {len(self.outputs)}"
            )

    @staticmethod
    def _format_expected(expected: set[int]) -> str:
        ordered = sorted(expected)
        if len(ordered) == 1:
            return str(ordered[0])
        return "one of " + ", ".join(str(value) for value in ordered)

    def _lookup_value(
        self, values: Mapping[str, Value], value_id: str, role: str
    ) -> Value:
        try:
            return values[value_id]
        except KeyError as exc:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} references unknown {role} value '{value_id}'"
            ) from exc

    def _input_values(self, values: Mapping[str, Value]) -> list[Value]:
        return [
            self._lookup_value(values, value_id, "input") for value_id in self.inputs
        ]

    def _output_value(self, values: Mapping[str, Value]) -> Value:
        self._require_output_count(1)
        return self._lookup_value(values, self.outputs[0], "output")

    def _require_tensor(self, value: Value, label: str) -> None:
        if value.vtype is not ValueType.TENSOR:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} expects {label} to be a tensor, "
                f"got {value.vtype.value}"
            )

    def _require_scalar_or_tensor(self, value: Value, label: str) -> None:
        if value.vtype not in (ValueType.SCALAR, ValueType.TENSOR):
            raise InvalidOperatorInstanceError(
                f"{self.op_type} expects {label} to be a scalar or tensor, "
                f"got {value.vtype.value}"
            )

    def _require_same_dtype(self, values: Sequence[Value]) -> None:
        if not values:
            return

        first_dtype = values[0].dtype
        for value in values[1:]:
            if value.dtype != first_dtype:
                raise InvalidOperatorInstanceError(
                    f"{self.op_type} requires all values to share dtype {first_dtype}"
                )

    def _resolve_axis(self, axis: int, rank: int, attr_name: str = "axis") -> int:
        if not isinstance(axis, int):
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires '{attr_name}' to be an integer"
            )
        normalized = axis if axis >= 0 else rank + axis
        if normalized < 0 or normalized >= rank:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} axis {axis} is out of range for rank {rank}"
            )
        return normalized

    def _require_int_attr(self, name: str) -> int:
        value = self.attrs.get(name)
        if not isinstance(value, int):
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires '{name}' to be of type int"
            )
        return value

    def _optional_int_attr(self, name: str, default: int) -> int:
        value = self.attrs.get(name, default)
        if not isinstance(value, int):
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires '{name}' to be of type int"
            )
        return value

    def _optional_bool_attr(self, name: str, default: bool) -> bool:
        value = self.attrs.get(name, default)
        if not isinstance(value, bool):
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires '{name}' to be of type bool"
            )
        return value

    def _require_int_sequence_attr(self, name: str) -> list[int]:
        if name not in self.attrs:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires '{name}' in attrs"
            )
        value = self.attrs[name]
        if not _is_int_sequence(value):
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires '{name}' to be a sequence of integers"
            )
        return list(value)

    def _match_output(
        self,
        values: Mapping[str, Value],
        *,
        shape: Sequence[int],
        axes: Sequence[str],
        dtype: str,
        vtype: ValueType,
    ) -> Value:
        output = self._output_value(values)
        if output.shape != list(shape):
            raise InvalidOperatorInstanceError(
                f"{self.op_type} expects output shape {list(shape)}, got {output.shape}"
            )
        if output.axes != list(axes):
            raise InvalidOperatorInstanceError(
                f"{self.op_type} expects output axes {list(axes)}, got {output.axes}"
            )
        if output.dtype != dtype:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} expects output dtype {dtype}, got {output.dtype}"
            )
        if output.vtype is not vtype:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} expects output value type {vtype.value}, "
                f"got {output.vtype.value}"
            )
        return output

    def _output_work(self, values: Mapping[str, Value]) -> int:
        return _shape_product(self._output_value(values).shape)


class UnaryElementwiseOperator(BuiltinOperator):
    OP_TYPE = "_UnaryElementwiseBase"

    def validate(self, values: Mapping[str, Value]) -> None:
        self._require_input_count(1)
        self._require_output_count(1)
        input_value = self._input_values(values)[0]
        self._require_scalar_or_tensor(input_value, "input[0]")
        output_type = (
            ValueType.SCALAR
            if input_value.vtype is ValueType.SCALAR
            else ValueType.TENSOR
        )
        self._match_output(
            values,
            shape=input_value.shape,
            axes=input_value.axes,
            dtype=input_value.dtype,
            vtype=output_type,
        )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        work = self._output_work(values)
        return FPGACost(
            latency_cycles=max(1, work),
            initiation_interval=1,
            lut=max(1, work),
            ff=max(1, work),
            metadata={"heuristic": "unary_elementwise"},
        )


class BinaryElementwiseOperator(BuiltinOperator):
    OP_TYPE = "_BinaryElementwiseBase"

    def validate(self, values: Mapping[str, Value]) -> None:
        self._require_input_count(2)
        self._require_output_count(1)
        lhs, rhs = self._input_values(values)
        self._require_scalar_or_tensor(lhs, "input[0]")
        self._require_scalar_or_tensor(rhs, "input[1]")
        self._require_same_dtype((lhs, rhs))

        if lhs.vtype is ValueType.SCALAR and rhs.vtype is ValueType.SCALAR:
            expected_shape = []
            expected_axes = []
            expected_type = ValueType.SCALAR
        elif lhs.vtype is ValueType.SCALAR:
            expected_shape = rhs.shape
            expected_axes = rhs.axes
            expected_type = rhs.vtype
        elif rhs.vtype is ValueType.SCALAR:
            expected_shape = lhs.shape
            expected_axes = lhs.axes
            expected_type = lhs.vtype
        elif lhs.shape == rhs.shape and lhs.axes == rhs.axes:
            expected_shape = lhs.shape
            expected_axes = lhs.axes
            expected_type = lhs.vtype
        else:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires matching tensor shapes or scalar broadcasting"
            )

        self._match_output(
            values,
            shape=expected_shape,
            axes=expected_axes,
            dtype=lhs.dtype,
            vtype=expected_type,
        )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        work = self._output_work(values)
        return FPGACost(
            latency_cycles=max(1, work),
            initiation_interval=1,
            lut=max(1, work),
            ff=max(1, work),
            metadata={"heuristic": "binary_elementwise"},
        )


class ReductionOperator(BuiltinOperator):
    OP_TYPE = "_ReductionBase"

    def _reduction_axes(self, input_value: Value) -> list[int]:
        axis_value = self.attrs.get("axis")
        if axis_value is None:
            return list(range(len(input_value.shape)))

        if isinstance(axis_value, int):
            return [self._resolve_axis(axis_value, len(input_value.shape))]
        if not _is_int_sequence(axis_value):
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires 'axis' to be an integer "
                f"or sequence of integers"
            )

        normalized = [
            self._resolve_axis(axis, len(input_value.shape)) for axis in axis_value
        ]
        if len(normalized) != len(set(normalized)):
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires unique reduction axes"
            )
        return sorted(normalized)

    def validate(self, values: Mapping[str, Value]) -> None:
        self._require_input_count(1)
        self._require_output_count(1)
        input_value = self._input_values(values)[0]
        self._require_tensor(input_value, "input[0]")

        reduction_axes = self._reduction_axes(input_value)
        keepdims = self._optional_bool_attr("keepdims", False)

        if keepdims:
            expected_shape = [
                1 if idx in reduction_axes else dim
                for idx, dim in enumerate(input_value.shape)
            ]
            expected_axes = list(input_value.axes)
            expected_type = ValueType.TENSOR
        else:
            expected_shape = [
                dim
                for idx, dim in enumerate(input_value.shape)
                if idx not in reduction_axes
            ]
            expected_axes = [
                axis_name
                for idx, axis_name in enumerate(input_value.axes)
                if idx not in reduction_axes
            ]
            expected_type = ValueType.TENSOR if expected_shape else ValueType.SCALAR

        self._match_output(
            values,
            shape=expected_shape,
            axes=expected_axes,
            dtype=input_value.dtype,
            vtype=expected_type,
        )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        input_value = self._input_values(values)[0]
        work = _shape_product(input_value.shape)
        return FPGACost(
            latency_cycles=max(1, work),
            initiation_interval=1,
            lut=max(1, work),
            ff=max(1, work // 2),
            metadata={"heuristic": "reduction"},
        )

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        context = super().hls_context(values)
        input_value = self._input_values(values)[0]
        output_value = self._output_value(values)
        input_size = _shape_product(input_value.shape)
        output_size = _shape_product(output_value.shape)
        reduction_size = input_size // max(1, output_size)
        context.update(
            {
                "input_size": input_size,
                "output_size": output_size,
                "reduction_size": reduction_size,
            }
        )
        return context


class Add(BinaryElementwiseOperator):
    OP_TYPE = "Add"


class Sub(BinaryElementwiseOperator):
    OP_TYPE = "Sub"


class Mul(BinaryElementwiseOperator):
    OP_TYPE = "Mul"

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        work = self._output_work(values)
        return FPGACost(
            latency_cycles=max(1, work + 1),
            initiation_interval=1,
            dsp=max(1, work),
            lut=max(1, work),
            ff=max(1, work),
            metadata={"heuristic": "binary_mul"},
        )


class Div(BinaryElementwiseOperator):
    OP_TYPE = "Div"

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        work = self._output_work(values)
        return FPGACost(
            latency_cycles=max(4, work * 4),
            initiation_interval=1,
            dsp=max(1, work),
            lut=max(8, work * 2),
            ff=max(8, work * 2),
            metadata={"heuristic": "binary_div"},
        )


class Sigmoid(UnaryElementwiseOperator):
    OP_TYPE = "Sigmoid"


class Tanh(UnaryElementwiseOperator):
    OP_TYPE = "Tanh"


class ReLU(UnaryElementwiseOperator):
    OP_TYPE = "ReLU"


class GELU(UnaryElementwiseOperator):
    OP_TYPE = "GELU"


class Softmax(BuiltinOperator):
    OP_TYPE = "Softmax"

    def validate(self, values: Mapping[str, Value]) -> None:
        self._require_input_count(1)
        self._require_output_count(1)
        input_value = self._input_values(values)[0]
        self._require_tensor(input_value, "input[0]")
        axis = self._optional_int_attr("axis", -1)
        self._resolve_axis(axis, len(input_value.shape))
        self._match_output(
            values,
            shape=input_value.shape,
            axes=input_value.axes,
            dtype=input_value.dtype,
            vtype=ValueType.TENSOR,
        )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        input_value = self._input_values(values)[0]
        work = _shape_product(input_value.shape)
        return FPGACost(
            latency_cycles=max(2, work * 2),
            initiation_interval=1,
            lut=max(2, work * 2),
            ff=max(2, work * 2),
            metadata={"heuristic": "softmax"},
        )

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        context = super().hls_context(values)
        input_value = self._input_values(values)[0]
        axis = self._resolve_axis(
            self._optional_int_attr("axis", -1),
            len(input_value.shape),
        )
        axis_size = input_value.shape[axis]
        outer_size = _shape_product(input_value.shape[:axis])
        inner_size = _shape_product(input_value.shape[axis + 1 :])
        context.update(
            {
                "axis_size": axis_size,
                "outer_size": outer_size,
                "inner_size": inner_size,
            }
        )
        return context


class Sum(ReductionOperator):
    OP_TYPE = "Sum"


class Mean(ReductionOperator):
    OP_TYPE = "Mean"


class Max(ReductionOperator):
    OP_TYPE = "Max"


class MatMul(BuiltinOperator):
    OP_TYPE = "MatMul"

    def validate(self, values: Mapping[str, Value]) -> None:
        self._require_input_count(2)
        self._require_output_count(1)
        lhs, rhs = self._input_values(values)
        self._require_tensor(lhs, "input[0]")
        self._require_tensor(rhs, "input[1]")
        self._require_same_dtype((lhs, rhs))

        if len(lhs.shape) != 2 or len(rhs.shape) != 2:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} currently supports rank-2 tensor inputs only"
            )
        if lhs.shape[1] != rhs.shape[0]:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires lhs.shape[1] == rhs.shape[0], got "
                f"{lhs.shape[1]} and {rhs.shape[0]}"
            )

        expected_shape = [lhs.shape[0], rhs.shape[1]]
        expected_axes = [lhs.axes[0], rhs.axes[1]]
        self._match_output(
            values,
            shape=expected_shape,
            axes=expected_axes,
            dtype=lhs.dtype,
            vtype=ValueType.TENSOR,
        )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        lhs, rhs = self._input_values(values)
        m_dim, k_dim = lhs.shape
        _, n_dim = rhs.shape
        work = m_dim * n_dim * k_dim
        return FPGACost(
            latency_cycles=max(1, work),
            initiation_interval=1,
            dsp=max(1, min(work, k_dim)),
            lut=max(1, m_dim * n_dim),
            ff=max(1, m_dim * n_dim),
            metadata={"heuristic": "matmul"},
        )

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        context = super().hls_context(values)
        lhs, rhs = self._input_values(values)
        context.update(
            {
                "m_dim": lhs.shape[0],
                "k_dim": lhs.shape[1],
                "n_dim": rhs.shape[1],
            }
        )
        return context


class Transpose(BuiltinOperator):
    OP_TYPE = "Transpose"

    def validate(self, values: Mapping[str, Value]) -> None:
        self._require_input_count(1)
        self._require_output_count(1)
        input_value = self._input_values(values)[0]
        self._require_tensor(input_value, "input[0]")
        perm = self._require_int_sequence_attr("perm")
        if len(perm) != len(input_value.shape):
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires 'perm' length {len(input_value.shape)}, "
                f"got {len(perm)}"
            )

        normalized_perm = [
            self._resolve_axis(axis, len(input_value.shape), "perm") for axis in perm
        ]
        if sorted(normalized_perm) != list(range(len(input_value.shape))):
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires 'perm' to be a permutation of axes"
            )

        expected_shape = [input_value.shape[axis] for axis in normalized_perm]
        expected_axes = [input_value.axes[axis] for axis in normalized_perm]
        self._match_output(
            values,
            shape=expected_shape,
            axes=expected_axes,
            dtype=input_value.dtype,
            vtype=ValueType.TENSOR,
        )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        work = self._output_work(values)
        return FPGACost(
            latency_cycles=max(1, work),
            initiation_interval=1,
            bram=max(1, math.ceil(work / 256)),
            lut=max(1, work // 2),
            ff=max(1, work // 2),
            metadata={"heuristic": "transpose"},
        )

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        context = super().hls_context(values)
        input_value = self._input_values(values)[0]
        perm = self._require_int_sequence_attr("perm")
        if len(input_value.shape) != 2 or perm != [1, 0]:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} HLS template currently supports rank-2 "
                "matrix transpose with perm [1, 0]"
            )
        context.update(
            {
                "rows": input_value.shape[0],
                "cols": input_value.shape[1],
            }
        )
        return context


class Reshape(BuiltinOperator):
    OP_TYPE = "Reshape"

    def validate(self, values: Mapping[str, Value]) -> None:
        self._require_input_count(1)
        self._require_output_count(1)
        input_value = self._input_values(values)[0]
        self._require_tensor(input_value, "input[0]")
        target_shape = self._require_int_sequence_attr("shape")
        if any(dim <= 0 for dim in target_shape):
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires 'shape' to contain positive integers"
            )
        if _shape_product(input_value.shape) != _shape_product(target_shape):
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires input and target shape "
                f"to preserve element count"
            )

        output = self._output_value(values)
        if output.shape != list(target_shape):
            raise InvalidOperatorInstanceError(
                f"{self.op_type} expects output shape {list(target_shape)}, "
                f"got {output.shape}"
            )
        if len(output.axes) != len(output.shape):
            raise InvalidOperatorInstanceError(
                f"{self.op_type} expects output axes length to match output rank"
            )
        if output.dtype != input_value.dtype or output.vtype is not ValueType.TENSOR:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} expects tensor output with dtype {input_value.dtype}"
            )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        work = self._output_work(values)
        return FPGACost(
            latency_cycles=1,
            initiation_interval=1,
            lut=max(1, work // 4),
            ff=max(1, work // 4),
            metadata={"heuristic": "reshape"},
        )

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        context = super().hls_context(values)
        context["num_elements"] = self._output_work(values)
        return context


class Concat(BuiltinOperator):
    OP_TYPE = "Concat"

    def validate(self, values: Mapping[str, Value]) -> None:
        self._require_input_count(range(2, 65))
        self._require_output_count(1)
        input_values = self._input_values(values)
        for idx, value in enumerate(input_values):
            self._require_tensor(value, f"input[{idx}]")

        self._require_same_dtype(input_values)
        base = input_values[0]
        axis = self._resolve_axis(self._require_int_attr("axis"), len(base.shape))

        expected_shape = list(base.shape)
        expected_shape[axis] = 0
        for value in input_values:
            if len(value.shape) != len(base.shape):
                raise InvalidOperatorInstanceError(
                    f"{self.op_type} requires inputs to have the same rank"
                )
            for dim_idx, dim in enumerate(value.shape):
                if dim_idx == axis:
                    expected_shape[axis] += dim
                elif dim != base.shape[dim_idx]:
                    raise InvalidOperatorInstanceError(
                        f"{self.op_type} requires non-concatenated dimensions to match"
                    )
            if value.axes != base.axes:
                raise InvalidOperatorInstanceError(
                    f"{self.op_type} requires inputs to have matching axes"
                )

        self._match_output(
            values,
            shape=expected_shape,
            axes=base.axes,
            dtype=base.dtype,
            vtype=ValueType.TENSOR,
        )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        work = self._output_work(values)
        return FPGACost(
            latency_cycles=max(1, work),
            initiation_interval=1,
            bram=max(1, math.ceil(work / 512)),
            lut=max(1, work // 2),
            ff=max(1, work // 2),
            metadata={"heuristic": "concat"},
        )

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        context = super().hls_context(values)
        input_values = self._input_values(values)
        context.update(
            {
                "num_inputs": len(input_values),
                "input_sizes_csv": ", ".join(
                    str(_shape_product(value.shape)) for value in input_values
                ),
                "output_size": self._output_work(values),
            }
        )
        return context


class Slice(BuiltinOperator):
    OP_TYPE = "Slice"

    def validate(self, values: Mapping[str, Value]) -> None:
        self._require_input_count(1)
        self._require_output_count(1)
        input_value = self._input_values(values)[0]
        self._require_tensor(input_value, "input[0]")
        axis = self._resolve_axis(
            self._require_int_attr("axis"),
            len(input_value.shape),
        )
        start = self._require_int_attr("start")
        end = self._require_int_attr("end")
        step = self._optional_int_attr("step", 1)
        if step <= 0:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires 'step' to be a positive integer"
            )
        if start < 0 or end <= start or end > input_value.shape[axis]:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires 0 <= start < end <= input dimension"
            )

        sliced_extent = math.ceil((end - start) / step)
        expected_shape = list(input_value.shape)
        expected_shape[axis] = sliced_extent
        self._match_output(
            values,
            shape=expected_shape,
            axes=input_value.axes,
            dtype=input_value.dtype,
            vtype=ValueType.TENSOR,
        )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        work = self._output_work(values)
        return FPGACost(
            latency_cycles=max(1, work),
            initiation_interval=1,
            lut=max(1, work // 2),
            ff=max(1, work // 2),
            metadata={"heuristic": "slice"},
        )

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        context = super().hls_context(values)
        input_value = self._input_values(values)[0]
        axis = self._resolve_axis(
            self._require_int_attr("axis"),
            len(input_value.shape),
        )
        if axis != 0 or len(input_value.shape) != 1:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} HLS template currently supports rank-1 slices "
                "along axis 0"
            )
        context.update(
            {
                "start": self._require_int_attr("start"),
                "end": self._require_int_attr("end"),
                "step": self._optional_int_attr("step", 1),
                "output_size": self._output_work(values),
            }
        )
        return context


class LayerNorm(BuiltinOperator):
    OP_TYPE = "LayerNorm"

    def validate(self, values: Mapping[str, Value]) -> None:
        self._require_input_count(1)
        self._require_output_count(1)
        input_value = self._input_values(values)[0]
        self._require_tensor(input_value, "input[0]")
        axis = self._optional_int_attr("axis", -1)
        self._resolve_axis(axis, len(input_value.shape))
        self._match_output(
            values,
            shape=input_value.shape,
            axes=input_value.axes,
            dtype=input_value.dtype,
            vtype=ValueType.TENSOR,
        )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        work = self._output_work(values)
        return FPGACost(
            latency_cycles=max(2, work * 2),
            initiation_interval=1,
            dsp=max(1, work // 4),
            lut=max(2, work * 2),
            ff=max(2, work * 2),
            metadata={"heuristic": "layer_norm"},
        )

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        context = super().hls_context(values)
        input_value = self._input_values(values)[0]
        axis = self._resolve_axis(
            self._optional_int_attr("axis", -1),
            len(input_value.shape),
        )
        normalized_size = _shape_product(input_value.shape[axis:])
        outer_size = _shape_product(input_value.shape[:axis])
        context.update(
            {
                "normalized_size": normalized_size,
                "outer_size": outer_size,
                "epsilon": self.attrs.get("epsilon", 1e-5),
            }
        )
        return context


class Conv1D(BuiltinOperator):
    OP_TYPE = "Conv1D"

    def validate(self, values: Mapping[str, Value]) -> None:
        self._require_input_count((2, 3))
        self._require_output_count(1)
        input_value, weight_value, *rest = self._input_values(values)
        self._require_tensor(input_value, "input[0]")
        self._require_tensor(weight_value, "input[1]")
        if len(input_value.shape) != 3 or len(weight_value.shape) != 3:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires rank-3 input and weight tensors"
            )
        if input_value.dtype != weight_value.dtype:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires matching dtypes for input and weight"
            )
        if input_value.shape[1] != weight_value.shape[1]:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires input channels to match weight channels"
            )

        if rest:
            bias_value = rest[0]
            if bias_value.vtype not in (ValueType.TENSOR, ValueType.SCALAR):
                raise InvalidOperatorInstanceError(
                    f"{self.op_type} expects optional bias to be scalar or tensor"
                )
            if bias_value.vtype is ValueType.TENSOR and bias_value.shape not in (
                [weight_value.shape[0]],
                [1, weight_value.shape[0], 1],
            ):
                raise InvalidOperatorInstanceError(
                    f"{self.op_type} expects bias shape to match output channels"
                )

        stride = self._optional_int_attr("stride", 1)
        padding = self._optional_int_attr("padding", 0)
        dilation = self._optional_int_attr("dilation", 1)
        for attr_name, attr_value in (
            ("stride", stride),
            ("padding", padding),
            ("dilation", dilation),
        ):
            if attr_value < 0:
                raise InvalidOperatorInstanceError(
                    f"{self.op_type} requires '{attr_name}' "
                    f"to be a non-negative integer"
                )
        if stride == 0 or dilation == 0:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires 'stride' and 'dilation' to be positive"
            )

        batch, _, input_length = input_value.shape
        out_channels, _, kernel_width = weight_value.shape
        numerator = input_length + (2 * padding) - (dilation * (kernel_width - 1)) - 1
        if numerator < 0:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} has invalid kernel/padding/dilation for input length"
            )
        output_length = math.floor(numerator / stride) + 1
        if output_length <= 0:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} produced non-positive output length"
            )

        self._match_output(
            values,
            shape=[batch, out_channels, output_length],
            axes=input_value.axes,
            dtype=input_value.dtype,
            vtype=ValueType.TENSOR,
        )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        input_value, weight_value, *_ = self._input_values(values)
        batch, _, _ = input_value.shape
        out_channels, in_channels, kernel_width = weight_value.shape
        output_length = self._output_value(values).shape[2]
        work = batch * out_channels * output_length * in_channels * kernel_width
        return FPGACost(
            latency_cycles=max(1, work),
            initiation_interval=1,
            dsp=max(1, in_channels * kernel_width),
            bram=max(1, math.ceil((out_channels * kernel_width) / 32)),
            lut=max(1, out_channels * output_length),
            ff=max(1, out_channels * output_length),
            metadata={"heuristic": "conv1d"},
        )

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        context = super().hls_context(values)
        input_value, weight_value, *rest = self._input_values(values)
        batch, in_channels, input_length = input_value.shape
        out_channels, _, kernel_width = weight_value.shape
        output_length = self._output_value(values).shape[2]
        context.update(
            {
                "batch": batch,
                "in_channels": in_channels,
                "input_length": input_length,
                "out_channels": out_channels,
                "kernel_width": kernel_width,
                "output_length": output_length,
                "stride": self._optional_int_attr("stride", 1),
                "padding": self._optional_int_attr("padding", 0),
                "dilation": self._optional_int_attr("dilation", 1),
                "has_bias": _cpp_bool(bool(rest)),
            }
        )
        return context


class Pad(BuiltinOperator):
    OP_TYPE = "Pad"

    def validate(self, values: Mapping[str, Value]) -> None:
        self._require_input_count(1)
        self._require_output_count(1)
        input_value = self._input_values(values)[0]
        self._require_tensor(input_value, "input[0]")
        pads = self._require_int_sequence_attr("pads")
        if len(pads) != len(input_value.shape) * 2:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires 'pads' length {len(input_value.shape) * 2}"
            )
        if any(pad < 0 for pad in pads):
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires all pad values to be non-negative integers"
            )

        expected_shape = []
        rank = len(input_value.shape)
        for idx, dim in enumerate(input_value.shape):
            expected_shape.append(dim + pads[idx] + pads[idx + rank])

        self._match_output(
            values,
            shape=expected_shape,
            axes=input_value.axes,
            dtype=input_value.dtype,
            vtype=ValueType.TENSOR,
        )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        work = self._output_work(values)
        return FPGACost(
            latency_cycles=max(1, work),
            initiation_interval=1,
            bram=max(1, math.ceil(work / 512)),
            lut=max(1, work // 2),
            ff=max(1, work // 2),
            metadata={"heuristic": "pad"},
        )

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        context = super().hls_context(values)
        input_value = self._input_values(values)[0]
        if len(input_value.shape) != 1:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} HLS template currently supports rank-1 padding"
            )
        pads = self._require_int_sequence_attr("pads")
        context.update(
            {
                "input_size": _shape_product(input_value.shape),
                "output_size": self._output_work(values),
                "pad_before": pads[0] if pads else 0,
                "pad_after": pads[len(pads) // 2] if pads else 0,
            }
        )
        return context


class Shift(BuiltinOperator):
    OP_TYPE = "Shift"

    def validate(self, values: Mapping[str, Value]) -> None:
        self._require_input_count(1)
        self._require_output_count(1)
        input_value = self._input_values(values)[0]
        self._require_tensor(input_value, "input[0]")
        axis = self._require_int_attr("axis")
        self._resolve_axis(axis, len(input_value.shape))
        amount = self._require_int_attr("amount")
        if amount == 0:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires non-zero 'amount'"
            )
        self._match_output(
            values,
            shape=input_value.shape,
            axes=input_value.axes,
            dtype=input_value.dtype,
            vtype=ValueType.TENSOR,
        )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        work = self._output_work(values)
        return FPGACost(
            latency_cycles=max(1, work),
            initiation_interval=1,
            bram=max(1, math.ceil(work / 256)),
            lut=max(1, work // 2),
            ff=max(1, work // 2),
            metadata={"heuristic": "shift"},
        )

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        context = super().hls_context(values)
        context.update(
            {
                "amount": self._require_int_attr("amount"),
                "output_size": self._output_work(values),
            }
        )
        return context


class LSTM(BuiltinOperator):
    OP_TYPE = "LSTM"

    def validate(self, values: Mapping[str, Value]) -> None:
        self._require_input_count(range(3, 9))
        self._require_output_count(range(0, 4))
        input_values = self._input_values(values)
        x = input_values[0]
        w = input_values[1]
        r = input_values[2]

        self._require_tensor(x, "input[0] (X)")
        self._require_tensor(w, "input[1] (W)")
        self._require_tensor(r, "input[2] (R)")

        if len(x.shape) != 3:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} expects rank-3 input X [seq_len, batch, input_size]"
            )
        seq_len, batch, input_size = x.shape

        direction = self.attrs.get("direction", "forward")
        num_directions = 2 if direction == "bidirectional" else 1
        hidden_size = self._require_int_attr("hidden_size")

        if w.shape != [num_directions, 4 * hidden_size, input_size]:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} expects weight W shape "
                f"[{num_directions}, {4 * hidden_size}, {input_size}], "
                f"got {w.shape}"
            )

        if r.shape != [num_directions, 4 * hidden_size, hidden_size]:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} expects recurrence R shape "
                f"[{num_directions}, {4 * hidden_size}, {hidden_size}], "
                f"got {r.shape}"
            )

        if len(input_values) > 3:
            bias = input_values[3]
            self._require_tensor(bias, "input[3] (B)")
            expected_bias_shape = [num_directions, 8 * hidden_size]
            if bias.shape != expected_bias_shape:
                raise InvalidOperatorInstanceError(
                    f"{self.op_type} expects bias B shape {expected_bias_shape}, "
                    f"got {bias.shape}"
                )

        # Validate outputs
        output_values = [
            self._lookup_value(values, value_id, "output") for value_id in self.outputs
        ]
        if output_values:
            y = output_values[0]
            if y.shape != [seq_len, num_directions, batch, hidden_size]:
                raise InvalidOperatorInstanceError(
                    f"{self.op_type} expects output Y shape "
                    f"[{seq_len}, {num_directions}, {batch}, {hidden_size}], "
                    f"got {y.shape}"
                )

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        input_value = self._input_values(values)[0]
        seq_len, batch, input_size = input_value.shape
        hidden_size = self._require_int_attr("hidden_size")
        direction = self.attrs.get("direction", "forward")
        num_directions = 2 if direction == "bidirectional" else 1

        # Heuristic: LSTM is roughly 4 MatMuls per orientation per step
        work = (
            seq_len
            * batch
            * num_directions
            * 4
            * (input_size + hidden_size)
            * hidden_size
        )
        return FPGACost(
            latency_cycles=max(10, work // 4),
            initiation_interval=1,
            dsp=max(4, (input_size + hidden_size) // 16),
            lut=max(100, hidden_size * 10),
            ff=max(100, hidden_size * 10),
            metadata={"heuristic": "lstm"},
        )

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        context = super().hls_context(values)
        input_values = self._input_values(values)
        x, _, _ = input_values[:3]
        seq_len, batch, input_size = x.shape
        hidden_size = self._require_int_attr("hidden_size")
        direction = self.attrs.get("direction", "forward")
        if direction not in ("forward", "reverse", "bidirectional"):
            raise InvalidOperatorInstanceError(
                f"{self.op_type} requires direction to be forward, reverse, "
                "or bidirectional"
            )
        if len(input_values) > 4:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} HLS template currently supports X, W, R, "
                "and optional B inputs only"
            )
        if len(self.outputs) != 1:
            raise InvalidOperatorInstanceError(
                f"{self.op_type} HLS template currently emits the Y output only"
            )

        num_directions = 2 if direction == "bidirectional" else 1
        context.update(
            {
                "seq_len": seq_len,
                "batch": batch,
                "input_size": input_size,
                "hidden_size": hidden_size,
                "num_directions": num_directions,
                "has_bias": _cpp_bool(len(input_values) > 3),
                "reverse_direction": _cpp_bool(direction == "reverse"),
            }
        )
        return context


BUILTIN_OPERATORS = [
    MatMul,
    Add,
    Sub,
    Mul,
    Div,
    Transpose,
    Reshape,
    Concat,
    Slice,
    Sigmoid,
    Tanh,
    ReLU,
    GELU,
    Softmax,
    Sum,
    Mean,
    Max,
    LayerNorm,
    Conv1D,
    Pad,
    Shift,
    LSTM,
]

BUILTIN_OPERATOR_TYPES = [
    operator_cls.operator_type() for operator_cls in BUILTIN_OPERATORS
]


def register_builtin_operators(registry: OperatorRegistry) -> None:
    for operator_cls in BUILTIN_OPERATORS:
        registry.register(operator_cls)
