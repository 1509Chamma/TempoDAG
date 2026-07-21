"""NB19 hardware probe, fixed-point datapath: the decisive physical test.

The float sweep (LOG.md) proved the II-bound paradigm at the SCHEDULER
level (achieved II == algebraic floor) but the physical clock pinned at
~68.5 ns from the float LUT index+mux path. Fixed point is the mandated
enabling lever: integer bit-slice table indexing (no float clip/fptosi,
BRAM-friendly), and 1-cycle integer adds that let the scheduler actually
share. This emits an ap_fixed burst-loop GRU and its asserting TB.

Q-format: ap_fixed<W,I> datapath; accumulation in a wider type. The tanh
table is indexed by a bit-slice of the pre-activation (integer op). The
golden trace emulates the same fixed-point quantization so the TB gate is
a genuine fixed-point functional check (coarse; the rigorous per-op
certificate is the NB03 family, future emitter work). ASCII-only prints.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

H, F, N = 16, 4, 64
SEED = 0


def _cli(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return int(a.split("=")[1])
    return default


TARGET_II = _cli("ii", 12)
W_BITS = _cli("wbits", 18)      # total bits
I_BITS = _cli("ibits", 6)       # integer bits (incl sign) -> range +-2^(I-1)
LUT_N = _cli("lutn", 512)
LUT_RANGE = 4.0
FRAC = W_BITS - I_BITS
WS = Path(f"C:/tmp/iibound_fx_w{W_BITS}i{I_BITS}_ii{TARGET_II}_l{LUT_N}")

rng = np.random.default_rng(SEED)
f32 = np.float32
_SCALE = float(1 << FRAC)
_HI = (1 << (W_BITS - 1)) - 1
_LO = -(1 << (W_BITS - 1))


def qf(x):
    """Quantize to the Q(I).(FRAC) grid, round-to-nearest, saturate."""
    q = np.rint(np.asarray(x, dtype=np.float64) * _SCALE)
    q = np.clip(q, _LO, _HI)
    return q / _SCALE


def _lit(x):
    s = f"{float(x):.9g}"
    return s + ("f" if ("." in s or "e" in s or "inf" in s) else ".0f")


def _carr(name, a, ctype="fx"):
    dims = "".join(f"[{d}]" for d in a.shape)
    flat = ", ".join(_lit(v) for v in a.flatten())
    return f"static const {ctype} {name}{dims} = {{{flat}}};"


# weights on the Q grid
Wz, Wr, Wn = (qf(rng.standard_normal((H, F)) * 0.3) for _ in range(3))
Uz, Ur, Un = (qf(rng.standard_normal((H, H)) * 0.25) for _ in range(3))
bz, br, bn = (qf(rng.standard_normal(H) * 0.1) for _ in range(3))
Wo = qf(rng.standard_normal(H) * 0.4)
X = qf(rng.standard_normal((N, F)) * 0.5)

# tanh table on the Q grid, indexed by bit-slice of (pre + RANGE)
LUT_STEP = 2.0 * LUT_RANGE / LUT_N
TANH_TABLE = qf(np.tanh(-LUT_RANGE + (np.arange(LUT_N) + 0.5) * LUT_STEP))
_IDX_SCALE = LUT_N / (2 * LUT_RANGE)


def tanh_q(x):
    xc = np.clip(x, -LUT_RANGE, LUT_RANGE - 1e-9)
    idx = np.minimum(((xc + LUT_RANGE) * _IDX_SCALE).astype(np.int64),
                     LUT_N - 1)
    return TANH_TABLE[idx]


def sig_q(x):
    return qf(0.5 + 0.5 * tanh_q(qf(0.5 * x)))


def dot_q(w, v):
    """Fixed-order pairwise tree, quantizing at each add (matches HW)."""
    p = qf(w * v)
    while p.shape[-1] > 1:
        p = qf(p[..., 0::2] + p[..., 1::2])
    return p[..., 0]


# oracle (reset-after / PyTorch form), all ops on the Q grid
h = np.zeros(H)
golden = np.zeros(N)
for t in range(N):
    az = qf(np.array([dot_q(Uz[i], h) for i in range(H)])
            + np.array([dot_q(Wz[i], X[t]) for i in range(H)]) + bz)
    ar = qf(np.array([dot_q(Ur[i], h) for i in range(H)])
            + np.array([dot_q(Wr[i], X[t]) for i in range(H)]) + br)
    z, r = sig_q(az), sig_q(ar)
    unh = np.array([dot_q(Un[i], h) for i in range(H)])
    an = qf(qf(r * unh) + np.array([dot_q(Wn[i], X[t]) for i in range(H)]) + bn)
    n = tanh_q(an)
    h = qf(qf(qf(1.0 - z) * n) + qf(z * h))
    golden[t] = float(dot_q(Wo, h))

# coarse fixed-point functional gate (this probe measures TIMING/AREA;
# rigorous per-op certificate is NB03 family). ~1 LSB/op over ~49 ops x
# amplification, generous engineering slack:
TOL = max(0.02, 60.0 / _SCALE)


def emit():
    WS.mkdir(parents=True, exist_ok=True)
    fx = f"ap_fixed<{W_BITS}, {I_BITS}>"
    acc = f"ap_fixed<{W_BITS + 8}, {I_BITS + 4}>"
    idx_bits = max(1, (LUT_N - 1).bit_length())
    body = f"""// II-bound FIXED-POINT streaming GRU prototype (NB19 physical point).
#include "ap_fixed.h"
#include "ap_int.h"
typedef {fx} fx;
typedef {acc} acc_t;
{_carr("TANH_LUT", TANH_TABLE, "fx")}
static fx tanh_lut(acc_t x) {{
#pragma HLS INLINE
  const fx lo = -{LUT_RANGE}, hi = {LUT_RANGE};
  fx xc = (x < lo) ? lo : ((x > hi) ? hi : (fx)x);
  // integer bit-slice index: floor((xc + RANGE) * (LUT_N / 2RANGE))
  const acc_t scaled = (acc_t)(xc + hi) * (acc_t){_IDX_SCALE};
  ap_uint<{idx_bits}> idx = (ap_uint<{idx_bits}>)scaled;
  return TANH_LUT[idx];
}}
static fx sig_lut(acc_t x) {{
#pragma HLS INLINE
  const fx half = 0.5;
  return half + half * tanh_lut((acc_t)(half * (fx)x));
}}
static acc_t dot16(const fx w[16], const fx h[16]) {{
#pragma HLS INLINE
  acc_t p[16];
#pragma HLS ARRAY_PARTITION variable=p complete
  for (int k = 0; k < 16; ++k) {{
#pragma HLS UNROLL
    p[k] = (acc_t)(w[k] * h[k]);
  }}
  for (int s = 8; s > 0; s >>= 1) {{
#pragma HLS UNROLL
    for (int k = 0; k < s; ++k) {{
#pragma HLS UNROLL
      p[k] = p[k] + p[k + s];
    }}
  }}
  return p[0];
}}
static acc_t dot4(const fx w[4], const fx v[4]) {{
#pragma HLS INLINE
  return (acc_t)((w[0]*v[0] + w[1]*v[1]) + (w[2]*v[2] + w[3]*v[3]));
}}
{_carr("WZ", Wz)}
{_carr("WR", Wr)}
{_carr("WN", Wn)}
{_carr("UZ", Uz)}
{_carr("UR", Ur)}
{_carr("UN", Un)}
{_carr("BZ", bz)}
{_carr("BR", br)}
{_carr("BN", bn)}
{_carr("WO", Wo)}
void gru_stream(const fx x[{N}][{F}], fx y[{N}]) {{
#pragma HLS ARRAY_PARTITION variable=x complete dim=2
  static fx h[{H}];
#pragma HLS ARRAY_PARTITION variable=h complete
sample_loop:
  for (int t = 0; t < {N}; ++t) {{
#pragma HLS PIPELINE II={TARGET_II}
    fx xv[{F}];
#pragma HLS ARRAY_PARTITION variable=xv complete
    for (int j = 0; j < {F}; ++j) {{
#pragma HLS UNROLL
      xv[j] = x[t][j];
    }}
    fx hn[{H}];
#pragma HLS ARRAY_PARTITION variable=hn complete
    for (int i = 0; i < {H}; ++i) {{
#pragma HLS UNROLL
      const acc_t az = dot16(UZ[i], h) + (dot4(WZ[i], xv) + (acc_t)BZ[i]);
      const acc_t ar = dot16(UR[i], h) + (dot4(WR[i], xv) + (acc_t)BR[i]);
      const fx z = sig_lut(az);
      const fx r = sig_lut(ar);
      const acc_t an = (acc_t)(r * (fx)dot16(UN[i], h))
                     + (dot4(WN[i], xv) + (acc_t)BN[i]);
      const fx n = tanh_lut(an);
      hn[i] = (fx)(((fx)1 - z) * n) + (fx)(z * h[i]);
    }}
    acc_t a[{H}];
#pragma HLS ARRAY_PARTITION variable=a complete
    for (int i = 0; i < {H}; ++i) {{
#pragma HLS UNROLL
      h[i] = hn[i];
      a[i] = (acc_t)(WO[i] * hn[i]);
    }}
    for (int s = 8; s > 0; s >>= 1) {{
#pragma HLS UNROLL
      for (int k = 0; k < s; ++k) {{
#pragma HLS UNROLL
        a[k] = a[k] + a[k + s];
      }}
    }}
    y[t] = (fx)a[0];
  }}
}}
"""
    (WS / "gru_stream.cpp").write_text(body, encoding="utf-8")

    xs = ",\n  ".join("{" + ", ".join(_lit(v) for v in X[t]) + "}"
                      for t in range(N))
    gs = ", ".join(_lit(v) for v in golden)
    tb = f"""// Asserting TB, fixed-point oracle (coarse Q-grid functional gate)
#include "ap_fixed.h"
#include <cmath>
#include <iostream>
typedef {fx} fx;
extern void gru_stream(const fx x[{N}][{F}], fx y[{N}]);
int main() {{
  static const fx x[{N}][{F}] = {{
  {xs}
  }};
  static const double golden[{N}] = {{{gs}}};
  fx y[{N}];
  gru_stream(x, y);
  int errors = 0;
  for (int t = 0; t < {N}; ++t) {{
    if (std::fabs((double)y[t] - golden[t]) > {TOL:g}) {{
      std::cerr << "MISMATCH t=" << t << " got " << (double)y[t]
                << " want " << golden[t] << std::endl;
      ++errors;
    }}
  }}
  std::cout << "II-bound FX prototype TB complete, errors=" << errors
            << std::endl;
  return errors == 0 ? 0 : 1;
}}
"""
    (WS / "gru_stream_tb.cpp").write_text(tb, encoding="utf-8")
    (WS / "hls.cfg").write_text(
        "part=xck26-sfvc784-2LV-c\n\n[hls]\nflow_target=vivado\n"
        f"syn.file={(WS / 'gru_stream.cpp').as_posix()}\n"
        "syn.top=gru_stream\n"
        f"tb.file={(WS / 'gru_stream_tb.cpp').as_posix()}\n"
        "clock=5.0ns\n", encoding="utf-8")
    print(f"emitted to {WS} (Q{I_BITS}.{FRAC}, tol {TOL:g}, II={TARGET_II}, "
          f"LUT {LUT_N}, clock 5ns)")


if __name__ == "__main__":
    emit()
