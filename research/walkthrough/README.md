# Walkthroughs

Three short demo notebooks, each built around one question and one honest
experiment. They come **pre-executed** — every number and output visible in
the notebook was produced by the code above it, so they can be read straight
through on GitHub without running anything.

| # | question | what it shows |
|---|---|---|
| [1](1_does_the_hardware_stay_accurate.ipynb) | Does a real model still work after being shrunk to fixed-point for the chip? | A GRU trained on the Mackey-Glass chaotic benchmark keeps **99% of its accuracy** in the chip's Q6.12 arithmetic, and still forecasts ~51× better than the naive baseline. |
| [2](2_why_it_is_fast.ipynb) | Where does 60 ns/sample come from? | The cost is the recurrence loop (~12 cycles), not the whole step (~440), because everything else overlaps across samples — and the overlap is proven bit-for-bit exact. |
| [3](3_attention_the_fpga_way.ipynb) | How do transformers fit on a streaming FPGA? | Re-associating the attention math turns it into a small-state recurrence that streams — an exact rewrite, with the accuracy trade measured honestly. |

## Re-running them

Everything is pure Python (NumPy, plus PyTorch for notebook 1); no FPGA
toolchain is involved:

```bash
python -m pip install -r ../requirements-research.txt
python -m nbconvert --to notebook --execute --inplace 2_why_it_is_fast.ipynb
```

Re-execution refreshes the outputs in place; seeds are pinned, so the
numbers reproduce.

## Going deeper

These three notebooks demonstrate the core ideas. The measured evidence
behind them lives in the docs — [benchmarks](../../docs/benchmarks.md),
[cost-model validation](../../docs/cost-model-validation.md), and
[accuracy retention](../../docs/accuracy-retention.md) — and every hardware
number can be regenerated through the harness under
[`../../experiments/`](../../experiments/).
