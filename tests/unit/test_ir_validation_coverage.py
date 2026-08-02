"""Coverage for core IR validation branches, operator base rules, and the
HLS template resolver's less-traveled paths."""

import sys
import types
from collections.abc import Mapping

import pytest

from tempo_dag.codegen.hls.generator import (
    HLSTemplateNotFoundError,
    resolve_hls_template_path,
)
from tempo_dag.device import FPGADevice, Memory, Resources
from tempo_dag.ir.graph import Graph
from tempo_dag.ir.op import (
    FPGACost,
    InvalidOperatorDefinitionError,
    InvalidOperatorInstanceError,
    Operator,
)
from tempo_dag.ir.validation import (
    IRValidationError,
    TopologyValidationError,
    validate_fpga_constraints,
    validate_operators,
    validate_topology,
)
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ops.builtins import Add


def _tensor(value_id, shape=(2,)):
    return Value(
        value_id=value_id,
        vtype=ValueType.TENSOR,
        dtype="float32",
        shape=list(shape),
        axes=[f"a{i}" for i in range(len(shape))],
    )


class _CostStubOperator(Operator):
    """Concrete operator with a configurable resource cost."""

    OP_TYPE = "CostStub"

    def __init__(self, op_id, inputs, outputs, cost=None, template="stub.cpp.tpl"):
        super().__init__(op_id, inputs, outputs)
        self._cost = cost or FPGACost(latency_cycles=1)
        self._template = template

    def validate(self, values: Mapping[str, Value]) -> None:
        return None

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        return self._cost

    def hls_template_path(self) -> str:
        return self._template

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        return {"op_id": self.op_id}


def _device(luts=100000, dsps=500, bram_36k=100):
    return FPGADevice(
        name="cov_device",
        vendor="CovVendor",
        part_number="COV-001",
        resources=Resources(luts=luts, ffs=200000, dsps=dsps, bram_36k=bram_36k),
        memory=Memory(on_chip_kb=4096, external_bandwidth_gbps=19.2),
    )


def test_validate_operators_allows_source_op_without_inputs():
    graph = Graph(
        values={"y": _tensor("y")},
        ops={"src": _CostStubOperator("src", [], ["y"])},
        graph_inputs=[],
        graph_outputs=["y"],
    )
    assert validate_operators(graph) is None


def test_validate_topology_flags_unreachable_operator():
    # The op consumes a value id absent from graph.values, so every listed
    # value is reachable but the operator itself is not.
    graph = Graph(
        values={"y": _tensor("y")},
        ops={"ghost_op": _CostStubOperator("ghost_op", ["missing"], ["y"])},
        graph_inputs=["y"],
        graph_outputs=[],
    )
    with pytest.raises(TopologyValidationError, match="Operator 'ghost_op'"):
        validate_topology(graph)


def test_validate_topology_flags_unreachable_graph_output():
    graph = Graph(
        values={"x": _tensor("x")},
        ops={},
        graph_inputs=["x"],
        graph_outputs=["phantom"],
    )
    with pytest.raises(TopologyValidationError, match="output 'phantom'"):
        validate_topology(graph)


def test_validate_fpga_constraints_flags_insufficient_dsps():
    cost = FPGACost(latency_cycles=1, dsp=64, bram=1, lut=1)
    graph = Graph(
        values={"y": _tensor("y")},
        ops={"op": _CostStubOperator("op", [], ["y"], cost=cost)},
        graph_inputs=[],
        graph_outputs=["y"],
    )
    with pytest.raises(IRValidationError, match="Insufficient DSPs"):
        validate_fpga_constraints(graph, _device(dsps=8))


def test_validate_fpga_constraints_flags_insufficient_bram():
    cost = FPGACost(latency_cycles=1, dsp=1, bram=64, lut=1)
    graph = Graph(
        values={"y": _tensor("y")},
        ops={"op": _CostStubOperator("op", [], ["y"], cost=cost)},
        graph_inputs=[],
        graph_outputs=["y"],
    )
    with pytest.raises(IRValidationError, match="Insufficient BRAM"):
        validate_fpga_constraints(graph, _device(bram_36k=8))


def test_operator_type_rejects_abstract_base_without_op_type():
    with pytest.raises(InvalidOperatorDefinitionError, match="non-empty OP_TYPE"):
        Operator.operator_type()


def test_operator_rejects_non_string_optional_name():
    with pytest.raises(InvalidOperatorInstanceError, match="name must be a string"):
        Add("add0", inputs=["l", "r"], outputs=["o"], name=7)


def test_resolve_hls_template_accepts_absolute_existing_path(tmp_path):
    template = tmp_path / "abs_template.cpp.tpl"
    template.write_text("// $op_id\n", encoding="utf-8")
    operator = _CostStubOperator("abs0", ["a"], ["b"], template=str(template))
    assert resolve_hls_template_path(operator) == template


def test_resolve_hls_template_handles_fileless_module(monkeypatch):
    module_name = "_tempo_dag_cov_fileless_mod"
    monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))

    class _FilelessOperator(_CostStubOperator):
        OP_TYPE = "FilelessStub"

    _FilelessOperator.__module__ = module_name
    operator = _FilelessOperator(
        "fileless0", ["a"], ["b"], template="definitely/not_here.cpp.tpl"
    )
    with pytest.raises(HLSTemplateNotFoundError, match="could not be resolved"):
        resolve_hls_template_path(operator)
