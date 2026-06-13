from tempo_dag.ir import (
    FPGACost,
    Graph,
    Operator,
    OperatorRegistry,
    Value,
    ValueType,
    get_default_registry,
)
from tempo_dag.ir.graph import Graph as GraphModuleAlias
from tempo_dag.ir.op import Operator as OperatorModuleAlias
from tempo_dag.ir.registry import (
    OperatorRegistry as RegistryModuleAlias,
)
from tempo_dag.ir.registry import (
    get_default_registry as get_default_registry_module_alias,
)
from tempo_dag.ir.value import Value as ValueModuleAlias
from tempo_dag.ir_graph import Graph as GraphCompatibilityAlias


def test_tempo_dag_ir_namespace_reexports_core_types() -> None:
    assert Graph is GraphModuleAlias
    assert Operator is OperatorModuleAlias
    assert OperatorRegistry is RegistryModuleAlias
    assert Value is ValueModuleAlias
    assert ValueType.TENSOR.value == "tensor"
    assert FPGACost is not None


def test_edge_lstm_ir_registry_alias_matches_default_registry() -> None:
    assert get_default_registry is get_default_registry_module_alias
    assert "Add" in get_default_registry().list_registered()


def test_edge_lstm_ir_graph_remains_compatible_alias() -> None:
    assert GraphCompatibilityAlias is Graph
