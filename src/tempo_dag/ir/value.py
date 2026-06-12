from __future__ import annotations

from enum import Enum


class ValueType(Enum):
    TENSOR = "tensor"
    STATE = "State"
    SCALAR = "scalar"


class Value:
    """
    Represents a value in the intermediate representation (IR).
    Args:
        value_id (str): The unique identifier for the value.
        vtype (ValueType): The type of the value (e.g., "tensor", "scalar")
        dtype (str): The data type of the value (e.g., "float32", "int64")
        shape (List[int]): The shape of the value if it is a tensor
        axes (List[str]): The names of the axes corresponding to the shape dimensions
        layout (Optional[str]): The memory layout of the tensor (e.g., "NCHW", "NHWC")
        quant (Optional[Dict[str, Any]]): Quantization parameters
            if applicable.
        producer_op_id (Optional[str]): The ID of the operation that produces this value
    """

    def __init__(
        self,
        value_id: str,
        vtype: ValueType,
        dtype: str,
        shape: list[int],
        axes: list[str],
        layout: str | None = None,
        quant: dict[str, float | int | str | None] | None = None,
        producer_op_id: str | None = None,
    ) -> None:
        self.value_id = value_id
        self.vtype = vtype
        self.dtype = dtype
        self.shape = shape
        self.axes = axes
        self.layout = layout
        self.quant = quant
        self.producer_op_id = producer_op_id

    def to_dict(self) -> dict[str, object]:
        """
        Convert the Value to a JSON-serializable dictionary.
        """
        return {
            "value_id": self.value_id,
            "vtype": self.vtype.value,
            "dtype": self.dtype,
            "shape": self.shape,
            "axes": self.axes,
            "layout": self.layout,
            "quant": self.quant,
            "producer_op_id": self.producer_op_id,
        }
