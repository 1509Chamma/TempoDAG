from __future__ import annotations

from typing import TYPE_CHECKING

from .op import Operator
from .value import Value

if TYPE_CHECKING:
    from .registry import OperatorRegistry


class Graph:
    """
    Represents the entire intermediate representation (IR) graph.

    Args:
        values (Dict[str, Value]): A dictionary mapping value IDs to Value objects
        ops (Dict[str, Operator]): A dictionary mapping operation IDs to Operator
            objects
        graph_inputs (List[str]): A list of value IDs that are the inputs to the graph
        graph_outputs (List[str]): A list of value IDs that are the outputs of the graph
        states (Optional[Dict[str, Value]]): An optional dictionary of state values
        registry (Optional[OperatorRegistry]): Registry used to construct operators
            from op_type names
    """

    def __init__(
        self,
        values: dict[str, Value],
        ops: dict[str, Operator],
        graph_inputs: list[str],
        graph_outputs: list[str],
        states: dict[str, Value] | None = None,
        registry: OperatorRegistry | None = None,
    ) -> None:
        self.values = dict(values)
        self.ops: dict[str, Operator] = {}
        self.graph_inputs = list(graph_inputs)
        self.graph_outputs = list(graph_outputs)
        self.states = dict(states) if states is not None else {}
        if registry is not None:
            self.registry = registry
        else:
            from .registry import get_default_registry

            self.registry = get_default_registry()

        for op_id, operator in ops.items():
            self._store_operator(op_id, operator)

    def add_operator(self, operator: Operator) -> Operator:
        """Insert an already-instantiated operator into the graph."""

        self._store_operator(operator.op_id, operator)
        return operator

    def create_operator(
        self,
        op_type: str,
        *,
        op_id: str,
        inputs: list[str],
        outputs: list[str],
        attrs: dict[str, object] | None = None,
        name: str | None = None,
        source_span: str | None = None,
    ) -> Operator:
        """
        Construct an operator from the graph registry and store it in the graph.
        """

        operator = self.registry.create(
            op_type,
            op_id=op_id,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            name=name,
            source_span=source_span,
        )
        return self.add_operator(operator)

    def to_dict(self) -> dict[str, object]:
        """
        Convert the entire graph to a JSON-serializable dictionary.
        """
        return {
            "values": {vid: v.to_dict() for vid, v in self.values.items()},
            "ops": {oid: o.to_dict() for oid, o in self.ops.items()},
            "graph_inputs": self.graph_inputs,
            "graph_outputs": self.graph_outputs,
            "states": {sid: s.to_dict() for sid, s in self.states.items()},
        }

    @staticmethod
    def _validate_operator(op_id: str, operator: Operator) -> None:
        if not isinstance(operator, Operator):
            raise TypeError("ops must contain Operator instances")
        if operator.op_id != op_id:
            raise ValueError(
                f"operator key '{op_id}' does not match "
                f"operator.op_id '{operator.op_id}'"
            )

    def _store_operator(self, op_id: str, operator: Operator) -> None:
        self._validate_operator(op_id, operator)
        self.ops[op_id] = operator
