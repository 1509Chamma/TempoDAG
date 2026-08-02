# Research

The science behind TempoDAG, in a form you can read and rerun.

## Start with the walkthroughs

The [`walkthrough/`](walkthrough/) folder has three short, plain-language
demo notebooks — each poses one question, runs one honest experiment, and
says what it means. They come pre-executed, so the outputs are visible right
on GitHub:

1. **[Does the hardware stay accurate?](walkthrough/1_does_the_hardware_stay_accurate.ipynb)**
   — a GRU trained on a chaotic benchmark keeps 99% of its accuracy through the
   fixed-point deploy.
2. **[Why it's fast](walkthrough/2_why_it_is_fast.ipynb)** — the cost is the
   recurrence loop, not the whole step, and the overlap is provably exact.
3. **[Attention the FPGA way](walkthrough/3_attention_the_fpga_way.ipynb)** — how
   softmax attention becomes a streaming recurrence.

To re-run one (pure Python, pinned seeds — the numbers reproduce):

```bash
python -m pip install -r requirements-research.txt
python -m nbconvert --to notebook --execute --inplace walkthrough/2_why_it_is_fast.ipynb
```

## The cost-model validation campaign

[`cost_model_validation.py`](cost_model_validation.py) is the harness behind
[docs/cost-model-validation.md](../docs/cost-model-validation.md): 26 designs
(hidden-size, input-width, and depth sweeps plus seven recurrent cells from
the literature), each with its initiation interval and DSP usage predicted
from the IR *before* synthesis runs. The raw prediction/measurement log is
[`results/cost_model_validation.jsonl`](results/cost_model_validation.jsonl).

```bash
python cost_model_validation.py --predict-only   # the static predictions, no FPGA tools
python cost_model_validation.py --report         # predicted vs. measured table
python cost_model_validation.py --only gru16     # re-run one design (needs Vitis)
```
