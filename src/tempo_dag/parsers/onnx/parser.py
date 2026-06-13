from __future__ import annotations

from typing import TYPE_CHECKING, Any

import onnx
import onnx.numpy_helper

from tempo_dag.ir.graph import Graph
from tempo_dag.ir.value import Value, ValueType

if TYPE_CHECKING:
    from tempo_dag.ir.registry import OperatorRegistry


class ONNXParser:
    """
    Parses ONNX models into the TempoDAG Intermediate Representation (IR).
    """

    def __init__(
        self,
        registry: OperatorRegistry | None = None,
        extra_op_mapping: dict[str, str] | None = None,
    ) -> None:
        self.registry = registry
        self.op_mapping = {
            "LSTM": "LSTM",
            "MatMul": "MatMul",
            "Gemm": "MatMul",
            "Add": "Add",
            "Sub": "Sub",
            "Mul": "Mul",
            "Div": "Div",
            "Relu": "ReLU",
            "Sigmoid": "Sigmoid",
            "Tanh": "Tanh",
            "Gelu": "GELU",
            "Softmax": "Softmax",
            "ReduceSum": "Sum",
            "ReduceMean": "Mean",
            "ReduceMax": "Max",
            "Transpose": "Transpose",
            "Reshape": "Reshape",
            "Concat": "Concat",
            "Slice": "Slice",
            "LayerNormalization": "LayerNorm",
            "Conv": "Conv1D",
            "Pad": "Pad",
        }
        if extra_op_mapping:
            self.op_mapping.update(extra_op_mapping)

    def register_op_mapping(self, onnx_op: str, ir_op: str) -> None:
        """Register a custom mapping from an ONNX operator name
        to an IR operator type."""
        self.op_mapping[onnx_op] = ir_op

    def _get_ir_op_type(self, onnx_op: str) -> str | None:
        """Resolve the IR operator type for a given ONNX operator name."""
        if onnx_op in self.op_mapping:
            return self.op_mapping[onnx_op]

        if self.registry:
            registered = self.registry.list_registered()
            for reg_op in registered:
                if reg_op.lower() == onnx_op.lower():
                    return reg_op

        return None

    def parse(self, model_path: str) -> Graph:
        """Load an ONNX model from disk and convert it to an IR Graph."""
        model = onnx.load(model_path)
        return self.parse_model(model)

    def parse_model(self, model: onnx.ModelProto) -> Graph:
        """Convert an in-memory ONNX ModelProto to an IR Graph."""
        onnx_graph = model.graph
        values: dict[str, Value] = {}
        ops: dict[str, Any] = (
            {}
        )  # Using Any here avoids tighter typing before operator mapping.

        initializers = {init.name for init in onnx_graph.initializer}
        graph_inputs = []
        for inp in onnx_graph.input:
            name = inp.name
            shape = self._get_onnx_shape(inp.type.tensor_type)
            dtype = self._get_onnx_dtype(inp.type.tensor_type.elem_type)
            values[name] = Value(
                value_id=name,
                vtype=ValueType.TENSOR,
                dtype=dtype,
                shape=shape,
                axes=[f"dim_{i}" for i in range(len(shape))],
            )
            if name not in initializers:
                graph_inputs.append(name)

        for init in onnx_graph.initializer:
            name = init.name
            shape = list(init.dims)
            dtype = self._get_onnx_dtype(init.data_type)
            values[name] = Value(
                value_id=name,
                vtype=ValueType.TENSOR,
                dtype=dtype,
                shape=shape,
                axes=[f"dim_{i}" for i in range(len(shape))],
            )

        for node in onnx_graph.node:
            onnx_op = node.op_type
            op_type = self._get_ir_op_type(onnx_op)
            if not op_type:
                raise ValueError(f"Unsupported ONNX operator: {onnx_op}")

            op_id = node.name or f"{op_type}_{len(ops)}"
            attrs = {
                attr.name: self._get_onnx_attribute(attr) for attr in node.attribute
            }

            # Special handling for Gemm: Map it to MatMul (+ Add if bias present)
            if onnx_op == "Gemm":
                matmul_out = f"{op_id}_matmul_out"
                if matmul_out not in values:
                    # Estimate shape from inputs... simplified for now
                    # Assuming lhs: [m, k], rhs: [k, n], out: [m, n]
                    lhs = values[node.input[0]]
                    rhs = values[node.input[1]]
                    values[matmul_out] = Value(
                        value_id=matmul_out,
                        vtype=ValueType.TENSOR,
                        dtype=lhs.dtype,
                        shape=[lhs.shape[0], rhs.shape[1]],
                        axes=[lhs.axes[0], rhs.axes[1]],
                    )

                ops[f"{op_id}_matmul"] = (
                    "MatMul",
                    [node.input[0], node.input[1]],
                    [matmul_out],
                    {},
                )

                if len(node.input) > 2:
                    ops[op_id] = (
                        "Add",
                        [matmul_out, node.input[2]],
                        list(node.output),
                        {},
                    )
                else:
                    ops.pop(f"{op_id}_matmul")
                    ops[op_id] = (
                        "MatMul",
                        [node.input[0], node.input[1]],
                        list(node.output),
                        {},
                    )
                continue

            # Special handling for LSTM hidden_size
            if op_type == "LSTM" and "hidden_size" not in attrs:
                if len(node.input) > 2:
                    r_name = node.input[2]
                    if r_name in values:
                        attrs["hidden_size"] = values[r_name].shape[2]

            ops[op_id] = (op_type, list(node.input), list(node.output), attrs)

        graph_outputs = [out.name for out in onnx_graph.output]
        for out in onnx_graph.output:
            if out.name not in values:
                shape = self._get_onnx_shape(out.type.tensor_type)
                dtype = self._get_onnx_dtype(out.type.tensor_type.elem_type)
                values[out.name] = Value(
                    value_id=out.name,
                    vtype=ValueType.TENSOR,
                    dtype=dtype,
                    shape=shape,
                    axes=[f"dim_{i}" for i in range(len(shape))],
                )

        # 5. Build IR Graph
        ir_graph = Graph(
            values=values,
            ops={},  # Will fill via create_operator
            graph_inputs=graph_inputs,
            graph_outputs=graph_outputs,
            registry=self.registry,
        )

        for op_id, (op_type, inputs, outputs, attrs) in ops.items():
            ir_graph.create_operator(
                op_type=op_type,
                op_id=op_id,
                inputs=inputs,
                outputs=outputs,
                attrs=attrs,
            )

        return ir_graph

    def _get_onnx_shape(self, tensor_type) -> list[int]:
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                # Unknown/Dynamic dimension: use 1 as a safe positive placeholder
                # for cost estimation and validation.
                shape.append(1)
        return shape

    def _get_onnx_dtype(self, elem_type: int) -> str:
        # Simplified mapping
        mapping: dict[int, str] = {
            onnx.TensorProto.FLOAT: "float32",
            onnx.TensorProto.DOUBLE: "float64",
            onnx.TensorProto.INT64: "int64",
            onnx.TensorProto.INT32: "int32",
        }
        return mapping.get(elem_type, "float32")

    def _get_onnx_attribute(self, attr: onnx.AttributeProto):
        if attr.HasField("f"):
            return attr.f
        if attr.HasField("i"):
            return attr.i
        if attr.HasField("s"):
            return attr.s.decode("utf-8")
        if attr.HasField("t"):
            return onnx.numpy_helper.to_array(attr.t)
        if attr.floats:
            return list(attr.floats)
        if attr.ints:
            return list(attr.ints)
        if attr.strings:
            return [s.decode("utf-8") for s in attr.strings]
        return None
