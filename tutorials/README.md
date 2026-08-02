# Tutorials

Hands-on notebooks for deploying a trained model through the compiler. The
[walkthroughs](../research/walkthrough/) explain *why* the approach works;
these show *how to use it* on a model of your own.

| # | notebook | what you'll do |
|---|---|---|
| 1 | [PyTorch GRU → RTL](01_pytorch_gru_to_rtl.py) | Train a GRU in PyTorch, rebuild it as a temporal process, prove numerical parity, emit the fixed-point HLS design, and (with Vitis installed) run C-sim → synthesis → RTL co-simulation. Includes the weight-layout maps for ONNX and Keras frontends. |
| 2 | [Custom operators](02_custom_operators.py) | Extend the compiler with your own operator — the IR node, the hardware C body, and the exact verification semantics registered as one object, so the testbench asserts your custom logic like any built-in. Demonstrated with a LeakyReLU on the recurrence path. |

The `.py` files are jupytext `py:percent` notebooks — read them top to bottom
as documents, run them as scripts, or convert to `.ipynb`:

```bash
python -m pip install -r ../requirements.txt jupytext
python 01_pytorch_gru_to_rtl.py            # runs end to end, ~30 s
jupytext --to ipynb 01_pytorch_gru_to_rtl.py
```

Steps 1–5 of each tutorial need only Python. The final RTL-simulation step
needs AMD Vitis (2024.2+); set `TUTORIAL_RUN_VITIS=1` and, if your install is
not in the default location, `VITIS_BIN`.

A note on the one detail that bites everyone: PyTorch, ONNX, and Keras all
store recurrent gate weights in different orders and layouts, and PyTorch's
GRU applies its reset gate differently than the textbook form. Tutorial 1
handles all of this explicitly and verifies the result numerically — if you
adapt it to your own model, keep the parity check. It is twenty lines and it
catches every layout mistake before you spend minutes on synthesis.

Generated output (C++, ONNX exports) lands in `tutorials/output/`, which is
not tracked.
