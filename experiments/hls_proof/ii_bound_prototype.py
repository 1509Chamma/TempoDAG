"""NB19 hardware probe: II-bound streaming GRU (PyTorch reset-after form).

Emits a handwritten burst-loop HLS kernel + asserting testbench into a
space-free workspace, ready for the v++ unified flow. The experiment
variable is the SAMPLE-LOOP II the scheduler achieves (floor 12 per NB19)
and whether flat tree-structured emission dodges the measured
full-unroll synthesis cliff (LOG.md: 4/4 one-hour blowups at H=16).

Emit only; the runner launches v++/vitis-run separately so synthesis can
be timeout-guarded and parallelized. ASCII-only prints.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

H, F, N = 16, 4, 64
SEED = 0
USE_LUT = "--lut" in sys.argv
def _cli(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return int(a.split("=")[1])
    return default
TARGET_II = _cli("ii", 12)
LUT_N = _cli("lutn", 4096)
SHARE = "--share" in sys.argv  # BIND_STORAGE bram LUT + ALLOCATION caps
_default_ws = "C:/tmp/iibound_proto2" if USE_LUT else "C:/tmp/iibound_proto"
WS = Path(f"C:/tmp/iibound_proto_ii{TARGET_II}_l{LUT_N}{'_sh' if SHARE else ''}"
          if (TARGET_II != 12 or LUT_N != 4096 or SHARE) else _default_ws)
LUT_RANGE = 4.0

rng = np.random.default_rng(SEED)
f32 = np.float32


def _lit(x: float) -> str:
    s = f"{float(x):.9g}"
    return s + ("f" if ("." in s or "e" in s or "inf" in s) else ".0f")


def _carr(name: str, a: np.ndarray) -> str:
    """static const float name[..] = {..}; row-major."""
    dims = "".join(f"[{d}]" for d in a.shape)
    flat = ", ".join(_lit(v) for v in a.flatten())
    return f"static const float {name}{dims} = {{{flat}}};"


# weights (LRU-ish scaling keeps the state bounded)
Wz, Wr, Wn = (rng.standard_normal((H, F)).astype(f32) * f32(0.3)
              for _ in range(3))
Uz, Ur, Un = (rng.standard_normal((H, H)).astype(f32) * f32(0.25)
              for _ in range(3))
bz, br, bn = (rng.standard_normal(H).astype(f32) * f32(0.1) for _ in range(3))
Wo = rng.standard_normal(H).astype(f32) * f32(0.4)
X = (rng.standard_normal((N, F)).astype(f32) * f32(0.5))


def sig(a):
    return (f32(1.0) / (f32(1.0) + np.exp(-a, dtype=f32))).astype(f32)


# tanh LUT (midpoint-sampled): the LUT DEFINES the activation (NB03/NB10
# principle) -- the oracle below uses identical semantics, so the TB gate
# only needs the reassociation budget, not an activation-error term.
LUT_STEP = 2.0 * LUT_RANGE / LUT_N
TANH_TABLE = np.tanh(-LUT_RANGE + (np.arange(LUT_N) + 0.5) * LUT_STEP
                     ).astype(f32)


def tanh_ref(x: np.ndarray) -> np.ndarray:
    if not USE_LUT:
        return np.tanh(x)
    xc = np.clip(x.astype(f32), f32(-LUT_RANGE), f32(LUT_RANGE))
    idx = np.minimum(((xc + f32(LUT_RANGE)) * f32(LUT_N / (2 * LUT_RANGE))
                      ).astype(np.int32), LUT_N - 1)
    return TANH_TABLE[idx]


def sig_ref(x: np.ndarray) -> np.ndarray:
    if not USE_LUT:
        return 1.0 / (1.0 + np.exp(-x))
    return (f32(0.5) + f32(0.5) * tanh_ref(f32(0.5) * x.astype(f32)))


# oracle (reset-after / PyTorch form), NB18-style budget gate
h64 = np.zeros(H, dtype=f32 if USE_LUT else np.float64)
golden = np.zeros(N)
for t in range(N):
    z = sig_ref(Uz.astype(h64.dtype) @ h64 + Wz.astype(h64.dtype) @ X[t].astype(h64.dtype) + bz.astype(h64.dtype))
    r = sig_ref(Ur.astype(h64.dtype) @ h64 + Wr.astype(h64.dtype) @ X[t].astype(h64.dtype) + br.astype(h64.dtype))
    n = tanh_ref(r * (Un.astype(h64.dtype) @ h64) + Wn.astype(h64.dtype) @ X[t].astype(h64.dtype) + bn.astype(h64.dtype))
    h64 = ((1.0 - z) * n + z * h64).astype(h64.dtype)
    golden[t] = float(Wo.astype(h64.dtype) @ h64)

# NB18 budget: n_ops=49 reductions/step, K=H+F+1 terms, M=2, rho=0.6, S=8
TOL = 8.0 * 49 * (H + F) * 1.19e-7 * 2.0 / 0.4


def emit() -> None:
    WS.mkdir(parents=True, exist_ok=True)
    mv_h = """
static float dot16(const float w[16], const float v[16]) {
#pragma HLS INLINE
  float p[16];
#pragma HLS ARRAY_PARTITION variable=p complete
  for (int k = 0; k < 16; ++k) {
#pragma HLS UNROLL
    p[k] = w[k] * v[k];
  }
  for (int s = 8; s > 0; s >>= 1) {
#pragma HLS UNROLL
    for (int k = 0; k < s; ++k) {
#pragma HLS UNROLL
      p[k] = p[k] + p[k + s];
    }
  }
  return p[0];
}
static float dot4(const float w[4], const float v[4]) {
#pragma HLS INLINE
  return (w[0]*v[0] + w[1]*v[1]) + (w[2]*v[2] + w[3]*v[3]);
}
"""
    act_defs = ""
    if USE_LUT:
        act_defs = f"""
{_carr("TANH_LUT", TANH_TABLE)}
static float tanh_lut(float x) {{
#pragma HLS BIND_STORAGE variable=TANH_LUT type=rom_2p impl=bram latency=2
#pragma HLS INLINE
  const float xc = (x < -{LUT_RANGE}f) ? -{LUT_RANGE}f
                 : ((x > {LUT_RANGE}f) ? {LUT_RANGE}f : x);
  int idx = (int)((xc + {LUT_RANGE}f) * {LUT_N / (2 * LUT_RANGE):.1f}f);
  if (idx > {LUT_N - 1}) idx = {LUT_N - 1};
  return TANH_LUT[idx];
}}
static float sig_lut(float x) {{
#pragma HLS INLINE
  return 0.5f + 0.5f * tanh_lut(0.5f * x);
}}
"""
    sig_expr = "sig_lut" if USE_LUT else "1.0f / (1.0f + std::exp(-(%s)))"
    tanh_expr = "tanh_lut" if USE_LUT else "std::tanh"
    body = f"""// II-bound streaming GRU prototype (NB19). Burst of {N} samples/call.
#include <cmath>
{act_defs}
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
{mv_h}
void gru_stream(const float x[{N}][{F}], float y[{N}]) {{
#pragma HLS ARRAY_PARTITION variable=x complete dim=2
  static float h[{H}];
#pragma HLS ARRAY_PARTITION variable=h complete
{"#pragma HLS ALLOCATION operation instances=fmul limit=80" if SHARE else ""}
{"#pragma HLS ALLOCATION operation instances=fadd limit=80" if SHARE else ""}
sample_loop:
  for (int t = 0; t < {N}; ++t) {{
#pragma HLS PIPELINE II={TARGET_II}
    float xv[{F}];
#pragma HLS ARRAY_PARTITION variable=xv complete
    for (int j = 0; j < {F}; ++j) {{
#pragma HLS UNROLL
      xv[j] = x[t][j];
    }}
    float hn[{H}];
#pragma HLS ARRAY_PARTITION variable=hn complete
    for (int i = 0; i < {H}; ++i) {{
#pragma HLS UNROLL
      const float az = dot16(UZ[i], h) + (dot4(WZ[i], xv) + BZ[i]);
      const float ar = dot16(UR[i], h) + (dot4(WR[i], xv) + BR[i]);
      const float z = {"sig_lut(az)" if USE_LUT else "1.0f / (1.0f + std::exp(-az))"};
      const float r = {"sig_lut(ar)" if USE_LUT else "1.0f / (1.0f + std::exp(-ar))"};
      const float n = {tanh_expr}(r * dot16(UN[i], h) + (dot4(WN[i], xv) + BN[i]));
      hn[i] = (1.0f - z) * n + z * h[i];
    }}
    float acc[{H}];
#pragma HLS ARRAY_PARTITION variable=acc complete
    for (int i = 0; i < {H}; ++i) {{
#pragma HLS UNROLL
      h[i] = hn[i];
      acc[i] = WO[i] * hn[i];
    }}
    for (int s = 8; s > 0; s >>= 1) {{
#pragma HLS UNROLL
      for (int k = 0; k < s; ++k) {{
#pragma HLS UNROLL
        acc[k] = acc[k] + acc[k + s];
      }}
    }}
    y[t] = acc[0];
  }}
}}
"""
    (WS / "gru_stream.cpp").write_text(body, encoding="utf-8")

    xs = ",\n  ".join(
        "{" + ", ".join(_lit(v) for v in X[t]) + "}" for t in range(N))
    gs = ", ".join(_lit(v) for v in golden)
    tb = f"""// Asserting TB for gru_stream (float64 oracle, NB18 budget gate)
#include <cmath>
#include <iostream>
extern void gru_stream(const float x[{N}][{F}], float y[{N}]);
int main() {{
  static const float x[{N}][{F}] = {{
  {xs}
  }};
  static const float golden[{N}] = {{{gs}}};
  float y[{N}];
  gru_stream(x, y);
  int errors = 0;
  for (int t = 0; t < {N}; ++t) {{
    if (std::fabs(y[t] - golden[t]) > {TOL:g}f) {{
      std::cerr << "MISMATCH t=" << t << " got " << y[t]
                << " want " << golden[t] << std::endl;
      ++errors;
    }}
  }}
  std::cout << "II-bound prototype TB complete, errors=" << errors
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
        "clock=6.1ns\n", encoding="utf-8")
    print(f"emitted to {WS} (tol {TOL:g}, target II={TARGET_II}, "
          f"clock 6.1ns per NB19 best binding)")


if __name__ == "__main__":
    emit()
