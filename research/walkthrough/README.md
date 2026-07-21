# Walkthroughs — start here

Three short, plain-language notebooks that explain what TempoDAG does and why
it works. Read these first. They are written to be understood, not to be
exhaustive — each one poses a question, runs one honest experiment, and states
what it means.

| # | question | what it shows |
|---|---|---|
| [1](1_does_the_hardware_stay_accurate.py) | Does a real model still work after we shrink it to fixed-point for the chip? | A GRU trained on the Mackey-Glass chaotic benchmark keeps **99% of its accuracy** in the chip's Q6.12 arithmetic, and still forecasts ~51× better than the naive baseline. |
| [2](2_why_it_is_fast.py) | Where does the 60 ns/sample come from? | You pay for the recurrence loop (~12 cycles), not the whole step (~440), because the rest overlaps across samples — and that overlap is proven bit-for-bit exact. |
| [3](3_attention_the_fpga_way.py) | How do transformers fit on a streaming FPGA? | Softmax attention is expensive and doesn't stream; re-associating the math turns it into a small-state recurrence that streams — an exact rewrite, with an honest accuracy trade. |

## How to read them

The `.py` files are the source of truth — they are literate notebooks in
[jupytext](https://jupytext.readthedocs.io) `py:percent` format, so the prose
lives in the comments and they read top-to-bottom. You can either:

- **read the `.py` directly** (all the explanation is there), or
- **run them** for the live numbers:

  ```bash
  python -m pip install -r ../requirements-research.txt jupytext
  python 1_does_the_hardware_stay_accurate.py      # or open as a notebook
  jupytext --to ipynb 1_does_the_hardware_stay_accurate.py
  ```

Everything is pure Python (PyTorch + NumPy); none of it needs the FPGA
toolchain.

## Going deeper

These three distil a larger investigation. To reproduce the measured hardware
numbers end to end, see the benchmark harness under
[`../../experiments/`](../../experiments/), which emits each design and runs it
through the Vitis verification ladder.
