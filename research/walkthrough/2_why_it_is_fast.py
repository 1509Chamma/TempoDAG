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
# # Why it's fast: pay for the loop, not the whole computation
#
# The headline number is 60 nanoseconds per sample for a GRU. Where does that
# come from? It's one idea, and it's worth understanding because it's the whole
# reason this project exists.
#
# ## The naive way to think about it
#
# A GRU step is a chunk of arithmetic: a few matrix multiplies, some gates, a
# blend. On our hardware that chunk takes about 440 clock cycles from start to
# finish. So the obvious conclusion is: one sample = 440 cycles = ~2200 ns, and
# to go faster you have to make the arithmetic smaller.
#
# **That conclusion is wrong, and seeing why is the key insight.**
#
# ## The assembly-line view
#
# Think of the GRU step as a factory assembly line. A car takes 8 hours to build
# start-to-finish, but a factory doesn't ship one car every 8 hours — it ships
# one every few minutes, because dozens of cars are on the line at once, each at
# a different station.
#
# The only thing that limits how *often* you can start a new car is the slowest
# single station, plus one hard constraint: **a step that needs the result of
# the previous car can't start until that result is ready.** For a GRU, almost
# all the arithmetic (the input matrix-multiplies, the output projection) does
# NOT depend on the previous sample — it can be "on the line" in parallel. Only
# the *recurrence* — the hidden state feeding back into itself — is a true
# dependency.
#
# So the real cost per sample is not the 440-cycle total. It's the length of the
# **feedback loop** — the handful of operations the hidden state must pass
# through before it's ready for the next sample. Everything else overlaps. That
# loop is about 12 cycles. Twelve cycles × 5 ns = **60 ns**. That's the number.
#
# Engineers call this the *initiation interval* (II): how often you can start a
# new sample. Our whole compiler is built to make II as small as the feedback
# loop allows, and then to make that loop physically fit in a fast clock.

# %%
import numpy as np

# %% [markdown]
# ## The one thing we have to prove: overlapping is safe
#
# The claim "the input work overlaps across samples" only holds if reordering it
# that way gives the *exact same answer* as doing each step start-to-finish. If
# it drifted even slightly, the speed-up would be a lie. So let's check it
# directly: run a GRU the plain sequential way, then run it the "pipelined" way
# where all the non-feedback work is hoisted out and done up front, and confirm
# the outputs are **bit-for-bit identical**.

# %%
H, F = 16, 4
rng = np.random.default_rng(0)
f32 = np.float32
# random small GRU, single precision so we can check bit-exactness
Wz, Wr, Wn = (rng.standard_normal((H, F)).astype(f32) * f32(0.3) for _ in range(3))
Uz, Ur, Un = (rng.standard_normal((H, H)).astype(f32) * f32(0.3) for _ in range(3))
bz, br, bn = (rng.standard_normal(H).astype(f32) * f32(0.1) for _ in range(3))
X = rng.standard_normal((256, F)).astype(f32)


def mv(W, v):                       # matrix-vector, fixed summation order
    return (W @ v).astype(f32)


def sig(a):
    return (f32(1) / (f32(1) + np.exp(-a, dtype=f32))).astype(f32)


def gru_step(h, xz, xr, xn):
    # xz/xr/xn are the input-side partials Wx@x + b. Whether they were computed
    # inside this step or hoisted out earlier changes NOTHING about the result.
    z = sig((mv(Uz, h) + xz).astype(f32))
    r = sig((mv(Ur, h) + xr).astype(f32))
    n = np.tanh((r * mv(Un, h) + xn).astype(f32), dtype=f32)
    return ((f32(1) - z) * n + z * h).astype(f32)


# Way 1: plain sequential -- compute the input partials inside each step.
h = np.zeros(H, f32)
seq = []
for t in range(256):
    seq.append(h := gru_step(h, mv(Wz, X[t]) + bz, mv(Wr, X[t]) + br,
                             mv(Wn, X[t]) + bn))

# Way 2: pipelined -- hoist ALL the input partials out of the loop first
# (this is what the hardware does: that work overlaps neighbouring samples).
XZ = np.stack([mv(Wz, x) + bz for x in X])
XR = np.stack([mv(Wr, x) + br for x in X])
XN = np.stack([mv(Wn, x) + bn for x in X])
h = np.zeros(H, f32)
pipe = [h := gru_step(h, XZ[t], XR[t], XN[t]) for t in range(256)]

seq_a, pipe_a = np.stack(seq), np.stack(pipe)
identical = np.array_equal(seq_a.view(np.uint32), pipe_a.view(np.uint32))
print(f"pipelined output identical to sequential, bit-for-bit: {identical}")
print(f"largest difference across 256 samples: {np.max(np.abs(seq_a - pipe_a))}")

# %% [markdown]
# ## What this buys, in one table
#
# Because the overlap is provably exact, the per-sample cost drops from the full
# step latency to the feedback-loop depth. Measured on the real compiler,
# co-simulation-verified on the KV260:
#
# | | full step latency | feedback loop (II) | per sample |
# |---|---|---|---|
# | how you'd naively count | ~440 cycles | — | ~2200 ns |
# | what actually limits throughput | — | 12 cycles | **60 ns** |
# | diagonal-linear SSM (shorter loop) | — | 4 cycles | **20 ns** |
#
# And there's a bonus the assembly-line picture makes obvious: this cost is the
# same *no matter how long the input history is*. A transformer that re-reads a
# 128-step window pays 128× more for a longer context; a streaming recurrence
# carries its state and pays the same 60 ns whether the history is 8 steps or
# 8000. That "window-independence" is why the lead over window-based tools grows
# with sequence length.
#
# ## The takeaway
#
# Going faster was never about doing less arithmetic. It was about noticing that
# only a tiny part of the arithmetic is actually on the critical path, proving
# the rest can overlap, and then building a compiler that lays it out that way
# and makes the small critical loop close a fast clock in fixed-point. That's
# the 60 ns.
