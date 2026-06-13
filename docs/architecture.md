# Architecture

## Summary

TempoDAG is currently a compiler foundation for sequence-model acceleration on
FPGA targets. The repository already covers model ingestion, IR construction,
primitive operator modelling, quantization helpers, representative-dataset
calibration, device metadata, and operator-level HLS template rendering.

The architecture is intentionally modular so higher-level lowering passes can be
added later without replacing the current core abstractions.

## Import Namespace

The repository uses a `src/` layout, which is only a filesystem convention.
That means `src/` should not appear in consumer imports.

For new public IR-facing imports, prefer:

- `tempo_dag.ir`
- `tempo_dag.ir.graph`
- `tempo_dag.ir.value`
- `tempo_dag.ir.registry`
- `tempo_dag.ir_temporal`

The current implementation still lives under `tempo_dag.*`, and that namespace
remains supported internally. `tempo_dag.ir_graph` is kept as a compatibility
alias for now.

## Compilation Flow Today

The current end-to-end flow is:

1. Parse a model from ONNX directly, or from PyTorch/TensorFlow by exporting to
   ONNX first.
2. Build an IR graph composed of typed values and registered operators.
3. Attach quantization metadata and optionally build representative datasets for
   calibration experiments.
4. Validate operators and render HLS source from operator templates.
5. Pair the graph with board metadata from the device registry when reasoning
   about hardware targets.

This is a strong baseline. The temporal IR layer adds process-level structure
above the existing same-timestep graph, but the project is still short of
graph-level scheduling, multi-operator lowering, or board deployment.

## Package Map

### `src/tempo_dag/ir`

The IR layer is the heart of the repo:

- `Value` carries identity, value type, dtype, shape, axes, layout, and
  quantization metadata.
- `Operator` is the abstract base for IR nodes and defines validation, FPGA
  cost estimation, and HLS hooks.
- `Graph` stores values and operators and can construct operators through a
  registry.
- `OperatorRegistry` manages runtime registration for built-in and custom ops.
- `validation.py` contains graph-level validation helpers.

This layer is already useful for structural modelling and testing independent of
any final backend.

### `src/tempo_dag/ir_temporal`

The temporal IR layer models streaming execution across timesteps while reusing
the existing `tempo_dag.ir.Graph` as the same-timestep kernel representation:

- `Process` groups clocks, kernels, state, buffers, and temporal edges.
- `Kernel` wraps an acyclic same-timestep graph.
- `StateSpec` describes persistent values such as hidden state, rolling
  buffers, and running statistics.
- `BufferSpec` describes bounded history storage such as delay lines and window
  buffers.
- `Edge0` represents same-timestep dependencies that must form a DAG.
- `EdgeDelta` represents positive-lag temporal dependencies and is the only
  legal way to express feedback cycles.

This makes temporal state explicit without replacing the current operator and
graph model.

### `src/tempo_dag/ops`

The built-in operator library currently covers a practical primitive set:

- Elementwise arithmetic: `Add`, `Sub`, `Mul`, `Div`
- Reshaping and routing: `Transpose`, `Reshape`, `Concat`, `Slice`, `Pad`,
  `Shift`
- Nonlinearities: `Sigmoid`, `Tanh`, `ReLU`, `GELU`, `Softmax`
- Reductions and normalization: `Sum`, `Mean`, `Max`, `LayerNorm`
- Tensor kernels: `MatMul`, `Conv1D`
- Recurrent placeholder coverage: `LSTM`

Each operator validates its inputs and outputs against the graph value
environment and exposes a coarse FPGA cost heuristic.

### `src/tempo_dag/parsers`

Model ingestion currently works through ONNX:

- `parsers/onnx/parser.py` loads ONNX models and maps nodes into the IR.
- `parsers/pytorch/parser.py` exports a PyTorch module to ONNX, then reuses the
  ONNX parser.
- `parsers/tensorflow/parser.py` exports a TensorFlow/Keras model to ONNX with
  `tf2onnx`, then reuses the ONNX parser.

This keeps the ingestion stack narrow and reduces the amount of
framework-specific lowering logic inside the repo.

### `src/tempo_dag/calibration`

The calibration package focuses on representative dataset selection and
distributional checks for quantization workflows:

- `create_representative_dataset(...)` samples a bounded subset from an input
  stream
- `compute_stats(...)` builds distribution summaries
- `compare_stats(...)` reports drift between the full dataset and the selected
  subset
- Sampling strategies currently include uniform, stratified temporal,
  regime-aware, and tail-aware modes

This is a useful bridge between model-side data behaviour and later
deployment-side quantization choices.

### `src/tempo_dag/codegen/hls`

The HLS codegen layer is currently template-driven:

- Operators point at a template path
- `resolve_hls_template_path(...)` resolves repo-local or module-local templates
- `render_operator_hls(...)` validates the operator and renders the template
  with `string.Template`

Today this is operator-scoped code generation, not a full graph scheduler or
backend flow.

### `src/tempo_dag/device`

The device layer provides structured board metadata:

- `FPGADevice` and nested dataclasses describe resources, memory, IO,
  capabilities, and policies
- `DeviceRegistry` loads JSON presets from `configs/devices/`
- Presets can be overridden and validated at runtime

This metadata is the natural place to anchor future scheduling and backend
decisions.

## Top-Level Assets

- `configs/devices/`: Example board presets
- `hls/operators/`: Operator HLS templates
- `tests/`: Unit and integration coverage
- `include/` and `CMakeLists.txt`: Minimal C++ scaffolding for future native
  integration
- `Dockerfile`: A reproducible Python 3.12-based development container

## Extension Points

The current design already exposes several useful seams:

- Register custom operators with a dedicated `OperatorRegistry`
- Ship operator-local HLS templates next to Python modules
- Extend ONNX operator mappings in the parser
- Build streaming processes with `tempo_dag.ir_temporal.Process`
- Add new sampling strategies under `tempo_dag.calibration`
- Add new device presets without changing Python code

## Current Boundaries

The most important architectural boundary to understand is that the repo is
currently stronger on front-end modelling than on final hardware emission.

In practice that means:

- IR construction is real and tested
- Temporal process scaffolding and same-timestep DAG validation are real and
  tested
- Operator validation and template rendering are real and tested
- Calibration utilities are real and tested
- End-to-end recurrent lowering, hardware scheduling, deployment packaging, and
  board execution are still future work

