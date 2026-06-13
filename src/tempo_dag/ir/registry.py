from __future__ import annotations

import inspect

from .op import InvalidOperatorDefinitionError, Operator, OperatorError


class OperatorRegistryError(OperatorError):
    """Base exception for registry failures."""


class DuplicateOperatorError(OperatorRegistryError):
    """Raised when an operator type is registered more than once."""


class UnknownOperatorError(OperatorRegistryError):
    """Raised when an operator type is requested but not registered."""


class OperatorRegistry:
    """Runtime registry for operator classes and instance construction."""

    def __init__(self) -> None:
        self._operators: dict[str, type[Operator]] = {}

    def register(self, operator_cls: type[Operator]) -> type[Operator]:
        if not inspect.isclass(operator_cls) or not issubclass(operator_cls, Operator):
            raise InvalidOperatorDefinitionError(
                "operator_cls must be a concrete Operator subclass"
            )
        if inspect.isabstract(operator_cls):
            raise InvalidOperatorDefinitionError(
                f"{operator_cls.__name__} must be concrete before registration"
            )

        op_type = operator_cls.operator_type()
        if op_type in self._operators:
            raise DuplicateOperatorError(
                f"operator type '{op_type}' is already registered"
            )

        self._operators[op_type] = operator_cls
        return operator_cls

    def get(self, op_type: str) -> type[Operator]:
        try:
            return self._operators[op_type]
        except KeyError as exc:
            raise UnknownOperatorError(
                f"operator type '{op_type}' is not registered"
            ) from exc

    def create(
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
        operator_cls = self.get(op_type)
        return operator_cls(
            op_id=op_id,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            name=name,
            source_span=source_span,
        )

    def list_registered(self) -> list[str]:
        return sorted(self._operators)


def _build_default_registry() -> OperatorRegistry:
    registry = OperatorRegistry()

    from tempo_dag.ops.builtins import register_builtin_operators

    register_builtin_operators(registry)
    return registry


_DEFAULT_REGISTRY_INSTANCE: OperatorRegistry | None = None


def get_default_registry() -> OperatorRegistry:
    """Lazily build and return the singleton default operator registry."""
    global _DEFAULT_REGISTRY_INSTANCE
    if _DEFAULT_REGISTRY_INSTANCE is None:
        _DEFAULT_REGISTRY_INSTANCE = _build_default_registry()
    return _DEFAULT_REGISTRY_INSTANCE
