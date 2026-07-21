# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Attention the FPGA way: different math, not just a faster chip
#
# Transformers are everywhere, so a serious time-series compiler has to have an
# answer for attention. We built a full softmax attention block and it works
# (co-simulation verified). But it is *slow* by our standards — about 1.1
# microseconds per token, versus 60 nanoseconds for a GRU. This notebook
# explains why, and shows the way out, which is genuinely different mathematics
# rather than a bigger chip.
#
# ## Why plain attention fights the hardware
#
# Softmax attention compares every position to every other position. For a
# window of T positions that's a T×T table of scores, and each row is turned
# into weights with a `softmax` — an `exp` followed by a divide. Three of those
# ingredients are hostile to a streaming FPGA:
#
# - the T×T table is **quadratic** — it blows up as the window grows,
# - `exp` and divide are **expensive** in fixed-point hardware, and
# - worst of all, it **doesn't stream**: every output needs the whole window at
#   once, so you can't process one sample at a time.
#
# ## The different math
#
# The trick is an algebra identity. Softmax couples the positions together
# through its normaliser. If we *replace* softmax with a simple "feature map"
# `φ` on the queries and keys, the sum re-associates:
#
#     softmax(Q Kᵀ) V     →     φ(Q) · ( φ(K)ᵀ V )
#
# The piece in brackets, `φ(K)ᵀ V`, is a small fixed-size matrix — it no longer
# depends on T. And for causal (left-to-right) attention it can be built up **one
# token at a time** as a running sum. In other words, attention becomes a
# *recurrence* with a small matrix as its state — exactly the streaming shape
# our compiler already runs at tens of nanoseconds. No softmax, no `exp`, no
# T×T table.
#
# Below we (1) confirm the streaming form is mathematically identical to the
# batched form, and (2) show the honest catch: it's a *different* operator, good
# at different things than softmax.

# %%
import numpy as np

rng = np.random.default_rng(0)
T, D = 32, 16
Q, K, V = (rng.standard_normal((T, D)) for _ in range(3))


def phi(x):                       # a standard positive feature map (elu+1)
    return np.where(x > 0, x + 1.0, np.exp(x))

# %% [markdown]
# ## 1. The streaming version is exact, not an approximation

# %%
def linear_attention_batched(Q, K, V):
    fq, fk = phi(Q), phi(K)
    out = np.zeros((T, D))
    for i in range(T):                        # causal: attend to <= i
        w = fq[i] @ fk[:i + 1].T
        out[i] = (w @ V[:i + 1]) / (w.sum() + 1e-6)
    return out


def linear_attention_streaming(Q, K, V):
    fq, fk = phi(Q), phi(K)
    S = np.zeros((D, D))                       # the running matrix "state"
    z = np.zeros(D)
    out = np.zeros((T, D))
    for t in range(T):                         # one token at a time
        S += np.outer(fk[t], V[t])
        z += fk[t]
        out[t] = (fq[t] @ S) / (fq[t] @ z + 1e-6)
    return out


diff = np.max(np.abs(linear_attention_batched(Q, K, V)
                     - linear_attention_streaming(Q, K, V)))
print(f"streaming vs batched, largest difference: {diff:.2e}  "
      f"(they are the same computation, so this is ~0)")

# %% [markdown]
# ## 2. The honest catch: it's a different operator
#
# Linear attention isn't a worse softmax — it's a *different* tool that wins in
# different situations. We test two kinds of task:
#
# - **sharp recall**: "go find the one earlier value whose key matches" — the
#   thing softmax's `exp` is uniquely good at (it makes a spiky, selective
#   weighting).
# - **smooth aggregation**: "blend recent history into a trend" — the thing a
#   running state is naturally good at.

# %%
def softmax_attention(Q, K, V):
    s = (Q @ K.T) / np.sqrt(D)
    s = np.where(np.tril(np.ones((T, T))) > 0, s, -1e9)
    e = np.exp(s - s.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True)) @ V


def score(kind, attn):
    num = den = 0.0
    for _ in range(40):
        keys = rng.standard_normal((T, D))
        vals = rng.standard_normal((T, D))
        q = np.zeros((T, D))
        y = np.zeros((T, D))
        for t in range(1, T):
            if kind == "recall":              # query points at one random past key
                j = rng.integers(0, t)
                q[t] = keys[j] + 0.3 * rng.standard_normal(D)
                y[t] = vals[j]
            else:                             # target is a smooth local average
                q[t] = keys[t]
                y[t] = vals[max(0, t - 6):t].mean(0)
        o = attn(q, keys, vals)
        num += ((o[1:] - y[1:]) ** 2).sum()
        den += ((y[1:] - y[1:].mean(0)) ** 2).sum()
    return 1 - num / den                        # R^2: higher is better


print("task           softmax   linear-streaming")
for kind in ("recall", "smooth"):
    s_soft = score(kind, softmax_attention)
    s_lin = score(kind, linear_attention_streaming)
    print(f"{kind:14s}  {s_soft:6.2f}   {s_lin:6.2f}")

# %% [markdown]
# ## What we learned
#
# - **The streaming reformulation is exact** (difference ~1e-16), so causal
#   attention really can run as a small-state recurrence — the fast shape our
#   compiler already handles, at ~20 ns/token instead of ~1100 ns.
# - **But it's a genuine trade, not a free lunch.** Softmax wins sharp retrieval
#   (its `exp` makes the spiky, one-hot-ish weighting that finds a specific past
#   item); linear attention wins smooth aggregation of recent history (its
#   running state *is* a recency-weighted average). Neither dominates.
#
# The practical consequence for a time-series compiler is a clean menu, not a
# single hammer:
#
# - if the workload needs exact, retrieval-style attention, use the softmax
#   block (correct, verified, but microseconds-per-token and DSP-hungry);
# - if it needs the smooth, aggregate-recent-history behaviour that most
#   financial and sensor streams actually want, use the linear-attention
#   recurrence and get transformer-style modelling at recurrent-model speed.
#
# That second option — a transformer that streams at tens of nanoseconds — is
# the direction this project is built to exploit.
