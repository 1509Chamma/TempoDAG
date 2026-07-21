"""Fixed-point transformer encoder-block emitter (PatchTST core).

Companion to temporal_fixedpoint_generator.py. Where that emits the
DEPENDENCY-bound recurrent burst loop, this emits the DSP-bound feed-forward
transformer block that NB24 de-risked: a single-head self-attention + FFN
encoder layer over a window of T patch tokens, pre-norm + residual, all in
the same ap_fixed datapath and integer-bitslice LUT machinery.

New LUT primitives vs the recurrent emitter: exp (softmax, args<=0 after the
row-max subtract), GELU, and rsqrt (LayerNorm). Matmuls are emitted as
[T,K]@[K,M] nested balanced-tree reductions. The block is pure feed-forward
(no loop-carried state), so it pipelines; II is set by resource sharing, not
a recurrence chain (NB24). The golden is computed in the SAME fixed-point
semantics (oracle-relative certificate).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TransformerConfig:
    t_patches: int = 8
    d_model: int = 16
    d_ff: int = 32
    # transformers need more integer headroom than the recurrent archs:
    # attention scores (Q.K over d dims) and FFN pre-activations reach ~+-40,
    # which WRAPS in Q6.12's +-32 range (ap_fixed AP_WRAP). Q8.16 -> +-128.
    w_bits: int = 24
    i_bits: int = 8
    lut_n: int = 1024
    clock_ns: float = 5.0
    part: str = "xck26-sfvc784-2LV-c"
    seed: int = 0

    @property
    def frac(self) -> int:
        return self.w_bits - self.i_bits

    @property
    def scale(self) -> float:
        return float(1 << self.frac)


def _q(x, cfg):
    hi = (1 << (cfg.w_bits - 1)) - 1
    return (
        np.clip(np.rint(np.asarray(x, np.float64) * cfg.scale), -hi - 1, hi) / cfg.scale
    )


def _tr(x, frac):
    s = float(1 << frac)
    return np.floor(np.asarray(x, np.float64) * s) / s


def _lit(x):
    s = f"{float(x):.9g}"
    return s + ("" if ("." in s or "e" in s or "inf" in s) else ".0")


def _nested(a):
    a = np.asarray(a)
    if a.ndim == 0:
        return _lit(float(a))
    return "{" + ", ".join(_nested(s) for s in a) + "}"


def _pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


# ---- fixed-point reference block (the oracle; mirrors the emitted C) --------
def _gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


class _Luts:
    def __init__(self, cfg):
        self.cfg = cfg
        self.exp = self._mk(np.exp, -8.0, 0.0)
        self.gelu = self._mk(_gelu, -6.0, 6.0)
        self.rsq = self._mk(lambda v: 1.0 / np.sqrt(v + 1e-5), 1e-4, 8.0)
        # reciprocal for the softmax normalizer (den in [~1, T+eps] after the
        # max-subtract): replaces T*T ap_fixed divisions with one LUT/row.
        self.recip = self._mk(lambda v: 1.0 / v, 0.5, cfg.t_patches + 1.0)

    def _mk(self, fn, lo, hi):
        n = self.cfg.lut_n
        step = (hi - lo) / n
        tab = _q(fn(lo + (np.arange(n) + 0.5) * step), self.cfg)
        return {"tab": tab, "lo": lo, "span": hi - lo}

    def apply(self, which, x):
        d = getattr(self, which)
        n = self.cfg.lut_n
        idx = np.floor(
            (np.clip(x, d["lo"], d["lo"] + d["span"] - 1e-9) - d["lo"]) / d["span"] * n
        ).astype(np.int64)
        return d["tab"][np.clip(idx, 0, n - 1)]


def _block_oracle(x, W, cfg, luts):
    """One encoder block in the emitter's exact ap_fixed semantics."""
    fr, ar = cfg.frac, cfg.frac + 4
    D = cfg.d_model

    def mm(a, b):
        out = np.zeros((a.shape[0], b.shape[1]))
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                out[i, j] = _tr(np.sum(_tr(a[i, :] * b[:, j], ar)), fr)
        return out

    def ln(z):
        mu = _tr(z.mean(-1, keepdims=True), fr)
        var = _tr(((z - mu) ** 2).mean(-1, keepdims=True), fr)
        return _tr((z - mu) * luts.apply("rsq", var), fr)

    def softmax(s):
        s = _tr(s - s.max(-1, keepdims=True), fr)
        e = luts.apply("exp", s)
        recip = luts.apply("recip", _tr(e.sum(-1, keepdims=True), fr))
        return _tr(e * recip, fr)

    inv = _q(1.0 / np.sqrt(D), cfg)
    h = ln(x)
    Q, K, V = mm(h, W["Wq"]), mm(h, W["Wk"]), mm(h, W["Wv"])
    Q = _tr(Q * inv, fr)  # fold 1/sqrt(d) into Q (smaller scores)
    A = softmax(mm(Q, K.T))
    x1 = _tr(x + mm(mm(A, V), W["Wo"]), fr)
    h2 = ln(x1)
    ff = mm(luts.apply("gelu", mm(h2, W["W1"])), W["W2"])
    return _tr(x1 + ff, fr)


def demo_block_weights(cfg):
    rng = np.random.default_rng(cfg.seed)
    D, DFF = cfg.d_model, cfg.d_ff
    return {
        "Wq": _q(rng.standard_normal((D, D)) * 0.25, cfg),
        "Wk": _q(rng.standard_normal((D, D)) * 0.25, cfg),
        "Wv": _q(rng.standard_normal((D, D)) * 0.25, cfg),
        "Wo": _q(rng.standard_normal((D, D)) * 0.25, cfg),
        "W1": _q(rng.standard_normal((D, DFF)) * 0.25, cfg),
        "W2": _q(rng.standard_normal((DFF, D)) * 0.25, cfg),
    }


# ---- C emission -------------------------------------------------------------
def _emit_lut(name, d, cfg):
    idx_bits = max(1, (cfg.lut_n - 1).bit_length())
    inv_span = cfg.lut_n / d["span"]
    return [
        f"static const fx {name}_TAB[{cfg.lut_n}] = {_nested(d['tab'])};",
        f"static fx {name}_lut(acc_t x) {{",
        "#pragma HLS INLINE",
        f"  const fx lo = {_lit(d['lo'])}, hi = {_lit(d['lo'] + d['span'])};",
        "  fx xc = (x < lo) ? lo : ((x > hi) ? hi : (fx)x);",
        f"  const acc_t s = (acc_t)(xc - lo) * (acc_t){_lit(inv_span)};",
        "  int i = (int)s;",
        f"  if (i > {cfg.lut_n - 1}) i = {cfg.lut_n - 1};",
        f"  return {name}_TAB[(ap_uint<{idx_bits}>)i];",
        "}",
    ]


def _emit_matmul(fn, K, M):
    """C for OUT[i][m] = sum_k A[i][k]*W[k][m], A is [.,K], W is [K,M].

    RESOURCE-SHARED form: the reduction is a pipelined sequential accumulate,
    NOT a fully unrolled tree. Transformers are DSP-bound (NB24): a fully
    unrolled 18k-MAC block explodes binding. This shares a small MAC pipeline
    across the whole block; II is set by sharing (the correct regime). The
    inner k-loop is pipelined so multiplies stream one/cycle.
    """
    Kp = _pow2(K)
    pad = (
        [
            f"      for (int k = {K}; k < {Kp}; ++k) {{",
            "#pragma HLS UNROLL",
            "        p[k] = (acc_t)0;",
            "      }",
        ]
        if Kp > K
        else []
    )
    return [
        f"  {fn}_i: for (int i = 0; i < ROWS; ++i) {{",
        f"    {fn}_m: for (int m = 0; m < {M}; ++m) {{",
        # pipeline over the OUTPUTS (i,m); tree-reduce only the K-wide
        # reduction (16 here) - tractable, unlike unrolling all i*m*k (which
        # explodes binding). Middle of the NB24 DSP/latency curve: ~K MACs,
        # one output/cycle throughput.
        "#pragma HLS PIPELINE II=1",
        f"      acc_t p[{Kp}];",
        "#pragma HLS ARRAY_PARTITION variable=p complete",
        f"      for (int k = 0; k < {K}; ++k) {{",
        "#pragma HLS UNROLL",
        "        p[k] = (acc_t)(A[i][k] * W[k][m]);",
        "      }",
        *pad,
        f"      for (int s = {Kp // 2}; s > 0; s >>= 1) {{",
        "#pragma HLS UNROLL",
        "        for (int kk = 0; kk < s; ++kk) p[kk] = p[kk] + p[kk + s];",
        "      }",
        "      OUT[i][m] = (fx)p[0];",
        "    }",
        "  }",
    ]


def render_transformer_artifact(cfg, weights):
    fx = f"ap_fixed<{cfg.w_bits}, {cfg.i_bits}>"
    acc = f"ap_fixed<{cfg.w_bits + 8}, {cfg.i_bits + 4}>"
    T, D, DFF = cfg.t_patches, cfg.d_model, cfg.d_ff
    luts = _Luts(cfg)
    top = "patchtst_block"

    pre = [
        f"// Fixed-point PatchTST encoder block T={T} D={D} DFF={DFF}",
        '#include "ap_fixed.h"',
        '#include "ap_int.h"',
        f"typedef {fx} fx;",
        f"typedef {acc} acc_t;",
    ]
    for nm, key in [
        ("EXP", "exp"),
        ("GELU", "gelu"),
        ("RSQ", "rsq"),
        ("RECIP", "recip"),
    ]:
        pre += _emit_lut(nm, getattr(luts, key), cfg)
    for wk, arr in weights.items():
        pre.append(
            f"static const fx {wk}[{arr.shape[0]}][{arr.shape[1]}] = "
            f"{_nested(arr)};"
        )
    inv = _lit(float(_q(1.0 / np.sqrt(D), cfg)))

    # helper macros via templated inline funcs would be cleaner; here we emit
    # explicit typed matmul helpers for each (ROWS,K,M) shape used.
    def matmul_fn(name, rows, K, M):
        body = _emit_matmul(f"{name}_loop", K, M)
        body = [ln.replace("ROWS", str(rows)) for ln in body]
        # NOT inlined -> Vitis shares one instance across all call sites
        # (Q/K/V/O reuse mm_proj), keeping the block DSP-bound and tractable.
        return (
            [
                f"static void {name}(const fx A[{rows}][{K}], "
                f"const fx W[{K}][{M}], fx OUT[{rows}][{M}]) {{"
            ]
            + body
            + ["}"]
        )

    helpers = []
    helpers += matmul_fn("mm_proj", T, D, D)  # h@Wq/Wk/Wv, ctx@Wo
    helpers += matmul_fn("mm_scores", T, D, T)  # Q@K^T (K^T is [D,T])
    helpers += matmul_fn("mm_ctx", T, T, D)  # A@V
    helpers += matmul_fn("mm_ff1", T, D, DFF)  # h2@W1
    helpers += matmul_fn("mm_ff2", T, DFF, D)  # gelu@W2

    body = [
        f"void {top}(const fx x[{T}][{D}], fx y[{T}][{D}]) {{",
        # resource-shared block (NB24: transformers are DSP-bound). No
        # top-level array partitioning / loop unrolling - a fully parallel
        # 18k-MAC block explodes binding; this shares MAC pipelines instead.
        f"  fx h[{T}][{D}]; fx Q[{T}][{D}]; fx K[{T}][{D}]; fx V[{T}][{D}];",
        f"  fx Kt[{D}][{T}]; fx scores[{T}][{T}]; fx A[{T}][{T}];",
        f"  fx ctx[{T}][{D}]; fx ao[{T}][{D}]; fx x1[{T}][{D}];",
        f"  fx h2[{T}][{D}]; fx ff1[{T}][{DFF}]; fx g[{T}][{DFF}]; "
        f"fx ff[{T}][{D}];",
        # --- LN1 ---
        f"  ln1: for (int i = 0; i < {T}; ++i) {{",
        "    acc_t s0 = 0;",
        f"    for (int k = 0; k < {D}; ++k) s0 += (acc_t)x[i][k];",
        f"    fx mu = (fx)(s0 * (acc_t){_lit(1.0 / D)});",
        "    acc_t sv = 0;",
        f"    for (int k = 0; k < {D}; ++k) {{ fx d = x[i][k]-mu; "
        f"sv += (acc_t)(d*d); }}",
        f"    fx var = (fx)(sv * (acc_t){_lit(1.0 / D)});",
        "    fx inv = RSQ_lut((acc_t)var);",
        f"    for (int k = 0; k < {D}; ++k) h[i][k] = (fx)((x[i][k]-mu)*inv);",
        "  }",
        "  mm_proj(h, Wq, Q); mm_proj(h, Wk, K); mm_proj(h, Wv, V);",
        # fold 1/sqrt(d) into Q (keeps scores small; done before Q.K)
        f"  for (int i = 0; i < {T}; ++i) for (int k = 0; k < {D}; ++k) "
        f"Q[i][k] = (fx)((acc_t)Q[i][k] * (acc_t){inv});",
        # --- K^T ---
        f"  for (int a = 0; a < {D}; ++a) for (int b = 0; b < {T}; ++b) "
        "Kt[a][b] = K[b][a];",
        "  mm_scores(Q, Kt, scores);",
        # --- softmax rows (Q already scaled) ---
        f"  smax: for (int i = 0; i < {T}; ++i) {{",
        "    fx mx = scores[i][0];",
        f"    for (int j = 1; j < {T}; ++j) "
        "if (scores[i][j] > mx) mx = scores[i][j];",
        "    acc_t den = 0; fx e[" + str(T) + "];",
        f"    for (int j = 0; j < {T}; ++j) {{ e[j] = EXP_lut("
        "(acc_t)(scores[i][j]-mx)); den += (acc_t)e[j]; }",
        "    fx rden = RECIP_lut((acc_t)den);"
        f"    for (int j = 0; j < {T}; ++j) "
        "A[i][j] = (fx)((acc_t)e[j] * (acc_t)rden);",
        "  }",
        "  mm_ctx(A, V, ctx);",
        "  mm_proj(ctx, Wo, ao);",
        f"  for (int i = 0; i < {T}; ++i) for (int k = 0; k < {D}; ++k) "
        "x1[i][k] = (fx)((acc_t)x[i][k] + (acc_t)ao[i][k]);",
        # --- LN2 ---
        f"  ln2: for (int i = 0; i < {T}; ++i) {{",
        "    acc_t s0 = 0;",
        f"    for (int k = 0; k < {D}; ++k) s0 += (acc_t)x1[i][k];",
        f"    fx mu = (fx)(s0 * (acc_t){_lit(1.0 / D)});",
        "    acc_t sv = 0;",
        f"    for (int k = 0; k < {D}; ++k) {{ fx d = x1[i][k]-mu; "
        "sv += (acc_t)(d*d); }",
        f"    fx var = (fx)(sv * (acc_t){_lit(1.0 / D)});",
        "    fx inv = RSQ_lut((acc_t)var);",
        f"    for (int k = 0; k < {D}; ++k) h2[i][k] = (fx)((x1[i][k]-mu)" "*inv);",
        "  }",
        "  mm_ff1(h2, W1, ff1);",
        f"  for (int i = 0; i < {T}; ++i) for (int k = 0; k < {DFF}; ++k) "
        "g[i][k] = GELU_lut((acc_t)ff1[i][k]);",
        "  mm_ff2(g, W2, ff);",
        f"  for (int i = 0; i < {T}; ++i) for (int k = 0; k < {D}; ++k) "
        "y[i][k] = (fx)((acc_t)x1[i][k] + (acc_t)ff[i][k]);",
        "}",
    ]
    dut = "\n".join(pre + [""] + helpers + [""] + body + [""])

    # golden from the oracle
    rng = np.random.default_rng(cfg.seed + 1)
    X = _q(rng.standard_normal((T, D)) * 0.5, cfg)
    golden = _block_oracle(X, weights, cfg, luts)
    tb = _render_tb(X, golden, cfg, top)
    return dut, tb, top, X, golden


def _render_tb(X, golden, cfg, top):
    T, D = cfg.t_patches, cfg.d_model
    tol = 64.0 / cfg.scale  # softmax/LN amplify (NB24: ~0.035); ~64 LSB gate
    xs = _nested(X)
    return "\n".join(
        [
            f"// Oracle-relative fixed-point TB for {top}",
            '#include "ap_fixed.h"',
            "#include <cmath>",
            "#include <iostream>",
            f"typedef ap_fixed<{cfg.w_bits}, {cfg.i_bits}> fx;",
            f"extern void {top}(const fx x[{T}][{D}], fx y[{T}][{D}]);",
            "int main() {",
            f"  static const fx x[{T}][{D}] = {xs};",
            f"  static const double golden[{T}][{D}] = {_nested(golden)};",
            f"  fx y[{T}][{D}];",
            f"  {top}(x, y);",
            "  int errors = 0;",
            f"  for (int i = 0; i < {T}; ++i) for (int k = 0; k < {D}; ++k)",
            f"    if (std::fabs((double)y[i][k] - golden[i][k]) > {tol:g}) {{",
            '      std::cerr << "MISMATCH " << i << "," << k << " got "'
            " << (double)y[i][k]",
            '                << " want " << golden[i][k] << std::endl; ++errors; }',
            '  std::cout << "PatchTST block TB complete, errors=" << errors'
            " << std::endl;",
            "  return errors == 0 ? 0 : 1;",
            "}",
            "",
        ]
    )


def write_transformer_bundle(output_dir, cfg=None, weights=None, stem="patchtst"):
    cfg = cfg or TransformerConfig()
    weights = weights if weights is not None else demo_block_weights(cfg)
    dut, tb, top, _X, _g = render_transformer_artifact(cfg, weights)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.cpp").write_text(dut, encoding="utf-8")
    (out / f"{stem}_tb.cpp").write_text(tb, encoding="utf-8")
    return {"top": top, "dut": out / f"{stem}.cpp", "tb": out / f"{stem}_tb.cpp"}


__all__ = [
    "TransformerConfig",
    "demo_block_weights",
    "render_transformer_artifact",
    "write_transformer_bundle",
]
