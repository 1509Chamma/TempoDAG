import torch
import torch.nn as nn

from tempo_dag.ir.graph import Graph
from tempo_dag.parsers.pytorch.parser import PyTorchParser


class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


def test_pytorch_parser_converts_simple_module():
    parser = PyTorchParser()
    model = SimpleModule()
    example_input = torch.randn(1, 3)

    graph = parser.parse_module(model, example_input)

    assert isinstance(graph, Graph)
    assert len(graph.graph_inputs) == 1
    assert len(graph.graph_outputs) == 1

    # Check if expected operators are present
    # Gemm in ONNX is converted to MatMul + Add by our parser
    ops_found = {op.operator_type() for op in graph.ops.values()}
    assert "MatMul" in ops_found
    assert "Add" in ops_found
    assert "ReLU" in ops_found
