# Research

The science behind TempoDAG, in a form you can read and rerun.

## Start with the walkthroughs

The [`walkthrough/`](walkthrough/) folder has three short, plain-language
notebooks — each poses one question, runs one honest experiment, and says what
it means. Read these first:

1. **[Does the hardware stay accurate?](walkthrough/1_does_the_hardware_stay_accurate.py)**
   — a GRU trained on a chaotic benchmark keeps 99% of its accuracy through the
   fixed-point deploy.
2. **[Why it's fast](walkthrough/2_why_it_is_fast.py)** — you pay for the
   recurrence loop, not the whole step, and the overlap is provably exact.
3. **[Attention the FPGA way](walkthrough/3_attention_the_fpga_way.py)** — how
   softmax attention becomes a streaming recurrence.

They are [jupytext](https://jupytext.readthedocs.io) `py:percent` files (the
`.py` is the source of truth; the prose lives in the comments), so you can read
them top-to-bottom or run them:

```bash
python -m pip install -r requirements-research.txt jupytext
python walkthrough/2_why_it_is_fast.py            # pure NumPy
python walkthrough/1_does_the_hardware_stay_accurate.py   # needs PyTorch
```

## `lab/`

Small reproducibility helpers (seed control, provenance stamping) shared by the
benchmark harness under [`../experiments/`](../experiments/).
