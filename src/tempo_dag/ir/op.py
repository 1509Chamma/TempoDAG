from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from .value import Value


class OperatorError(Exception):
    """Base exception for operator definition and instantiation failures."""


class InvalidOperatorDefinitionError(OperatorError):
    """Raised when a concrete operator class violates the base contract."""


class InvalidOperatorInstanceError(OperatorError):
    """Raised when an operator instance is created with invalid node data."""


@dataclass(frozen=True)
class FPGACost:
    """Represents a coarse FPGA resource estimate for an operator."""

    latency_cycles: int
    initiation_interval: int = 1
    dsp: int = 0
    bram: int = 0
    lut: int = 0
    ff: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "latency_cycles",
            "initiation_interval",
            "dsp",
            "bram",
            "lut",
            "ff",
        ):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")


class Operator(ABC):
    """
    Abstract base class for IR graph operators.

    Concrete subclasses must define `OP_TYPE` and implement validation,
    cost estimation, and HLS generation hooks.
    """

    OP_TYPE: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return

        op_type = getattr(cls, "OP_TYPE", None)
        if not isinstance(op_type, str) or not op_type.strip():
            raise InvalidOperatorDefinitionError(
                f"{cls.__name__} must define a non-empty OP_TYPE class attribute"
            )

    def __init__(
        self,
        op_id: str,
        inputs: Sequence[str],
        outputs: Sequence[str],
        attrs: dict[str, object] | None = None,
        name: str | None = None,
        source_span: str | None = None,
    ) -> None:
        self.op_id = self._validate_identifier("op_id", op_id)
        self.op_type = self.operator_type()
        self.inputs = self._validate_identifier_list("inputs", inputs)
        self.outputs = self._validate_identifier_list("outputs", outputs)
        self.attrs = self._validate_attrs(attrs)
        self.name = self._validate_optional_string("name", name)
        self.source_span = self._validate_optional_string("source_span", source_span)

    @classmethod
    def operator_type(cls) -> str:
        op_type = cls.OP_TYPE
        if not isinstance(op_type, str) or not op_type.strip():
            raise InvalidOperatorDefinitionError(
                f"{cls.__name__} must define a non-empty OP_TYPE class attribute"
            )
        return op_type

    def to_dict(self) -> dict[str, object]:
        """Convert the operator to a JSON-serializable dictionary."""

        return {
            "op_id": self.op_id,
            "op_type": self.op_type,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attrs": self.attrs,
            "name": self.name,
            "source_span": self.source_span,
        }

    @abstractmethod
    def validate(self, values: Mapping[str, Value]) -> None:
        """Validate the operator instance against the graph value environment."""

    @abstractmethod
    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        """Estimate FPGA implementation cost for this operator instance."""

    @abstractmethod
    def hls_template_path(self) -> str:
        """Return the path to the HLS template used for code generation."""

    @abstractmethod
    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        """Return the template context used to render HLS for this operator."""

    @staticmethod
    def _validate_identifier(field_name: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise InvalidOperatorInstanceError(
                f"{field_name} must be a non-empty string"
            )
        return value

    @classmethod
    def _validate_identifier_list(
        cls, field_name: str, values: Sequence[str]
    ) -> list[str]:
        if isinstance(values, str) or not isinstance(values, Sequence):
            raise InvalidOperatorInstanceError(
                f"{field_name} must be a sequence of non-empty strings"
            )

        normalized = [
            cls._validate_identifier(f"{field_name}[{idx}]", value)
            for idx, value in enumerate(values)
        ]
        return normalized

    @staticmethod
    def _validate_attrs(attrs: dict[str, object] | None) -> dict[str, object]:
        if attrs is None:
            return {}
        if not isinstance(attrs, dict):
            raise InvalidOperatorInstanceError("attrs must be a dictionary when set")
        return dict(attrs)

    @staticmethod
    def _validate_optional_string(field_name: str, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise InvalidOperatorInstanceError(
                f"{field_name} must be a string when set"
            )
        return value


# Backwards-compatible alias while Graph and other call sites migrate to Operator.
Op = Operator
