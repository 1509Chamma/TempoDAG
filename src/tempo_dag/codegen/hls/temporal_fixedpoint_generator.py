"""Fixed-point II-bound burst-loop emitter for temporal processes.

This is the compiler realization of the handwritten prototype
(experiments/hls_proof/ii_bound_fixedpoint.py) that RTL-cosim-verified at
II=12 / est-clk 3.64ns / 941 DSP on the KV260. It consumes the SAME
Process IR the float emitter does (temporal_generator.py) but emits the
proven shape:

  - an ap_fixed<W,I> datapath (accumulation in a wider acc_t),
  - weights BAKED as file-scope `static const fx` arrays (no TV files,
    dodging the Windows 512-handle cosim cap),
  - tanh/sigmoid via a baked table indexed by an INTEGER bit-slice of the
    pre-activation (no float clip/fptosi - the path that pinned the float
    prototype at 68.5ns),
  - each op inlined and fully unrolled: MatMul -> partial products +
    balanced adder tree per output lane; elementwise/activation -> unrolled
    lane loops; state -> static ROMs read/written each iteration,
  - all wrapped in a single `<pid>_stream(x[N][K], y[N])` with the
    per-sample body under `#pragma HLS PIPELINE II=<target>`.

The float per-step path in temporal_generator.py is untouched; this is a
separate, additive backend selected explicitly (benchmark --fixedpoint).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path

import numpy as np

from tempo_dag.codegen.hls.temporal_generator import (
    _elems,
    _shape_of,
    _state_bindings,
    _step_interface,
    _topological_ops,
)
from tempo_dag.ir_temporal import Process
from tempo_dag.verification.golden_trace import GoldenTrace

_BINARY = {"Add": "+", "Sub": "-", "Mul": "*", "Div": "/"}
_ACT = {"Tanh": "tanh_lut", "Sigmoid": "sig_lut", "ReLU": "relu_fx"}
_SUPPORTED = set(_BINARY) | set(_ACT) | {"MatMul"}


@dataclass(frozen=True)
class CustomFixedPointOp:
    """A user-defined elementwise operator for the fixed-point emitter.

    Registering one extends the emitter AND the verification oracle in a
    single object, so the oracle-relative certificate covers custom logic
    exactly like the built-ins. The contract:

    - `c_name`: the C function name; called per lane as `c_name((acc_t)x)`.
    - `c_body`: the full C function definition (may use the `fx` and
      `acc_t` typedefs), emitted into the design's prelude.
    - `semantics`: a NumPy callable mirroring the C function EXACTLY on the
      fixed-point grid (inputs arrive as float64 values on the fx grid).
      If the two disagree, C-simulation fails loudly - which is the point.
    """

    c_name: str
    c_body: str
    semantics: object  # callable: np.ndarray -> np.ndarray


@dataclass(frozen=True)
class FixedPointConfig:
    """Datapath + schedule knobs for the burst-loop emitter.

    Defaults are the RTL-verified prototype working point (Q6.12, 512-entry
    tanh table over [-4, 4], II=12, 64-sample burst, 5ns clock).
    `custom_ops` maps an IR op_type to a CustomFixedPointOp, extending the
    emitter and oracle together (see tutorials/02_custom_operators).
    """

    w_bits: int = 18
    i_bits: int = 6
    lut_n: int = 512
    lut_range: float = 4.0
    target_ii: int = 12
    burst: int = 64
    clock_ns: float = 5.0
    part: str = "xck26-sfvc784-2LV-c"
    custom_ops: dict = field(default_factory=dict)

    @property
    def frac(self) -> int:
        return self.w_bits - self.i_bits

    @property
    def scale(self) -> float:
        return float(1 << self.frac)


@dataclass(frozen=True)
class FixedPointArtifact:
    """Emitted fixed-point burst-loop bundle."""

    dut_hls: str
    testbench_hls: str
    top_name: str


def _q(x, cfg: FixedPointConfig):
    """Round-to-nearest, saturating quantization onto the Q-grid."""
    hi = (1 << (cfg.w_bits - 1)) - 1
    lo = -(1 << (cfg.w_bits - 1))
    q = np.rint(np.asarray(x, dtype=np.float64) * cfg.scale)
    return np.clip(q, lo, hi) / cfg.scale


def _lit(x: float) -> str:
    # 17 significant digits: exact round-trip for any double. Every emitted
    # value (inputs, weights, LUT entries, golden) is a dyadic Q-format grid
    # point, so this prints it exactly. At 9 digits (the old format) a
    # literal could land a hair on the wrong side of its grid point, and
    # ap_fixed's AP_TRN construction then floors it one LSB low - a silent
    # 1-LSB perturbation of ~a quarter of all negative constants, which a
    # recurrent state amplifies past the testbench gate.
    s = f"{float(x):.17g}"
    return s + ("" if ("." in s or "e" in s or "inf" in s) else ".0")


def _nested(arr: np.ndarray) -> str:
    if arr.ndim == 0:
        return _lit(float(arr))
    return "{" + ", ".join(_nested(sub) for sub in arr) + "}"


def _is_weight2d(shape: list[int]) -> bool:
    return len(shape) == 2 and shape[0] > 1 and shape[1] >= 1


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


class _FixedPointUnsupported(Exception):
    """Raised when a process is outside the burst-loop emitter's scope."""


def render_fixedpoint_burst_artifact(
    process: Process,
    golden_trace: GoldenTrace,
    parameters: dict[str, object],
    config: FixedPointConfig | None = None,
) -> FixedPointArtifact:
    cfg = config or FixedPointConfig()
    if len(process.kernels) != 1:
        raise _FixedPointUnsupported("exactly one kernel required")
    kernel = next(iter(process.kernels.values()))
    graph = kernel.graph
    supported = _SUPPORTED | set(cfg.custom_ops)
    for op in graph.ops.values():
        if op.op_type not in supported:
            raise _FixedPointUnsupported(f"op '{op.op_type}' not supported")

    stream_ins, params, outs = _step_interface(process, kernel)
    stream_vid = stream_ins[0]
    out_vid = outs[0]
    states = _state_bindings(process, kernel)
    state_read = {b["read"]: b for b in states}
    baked = {vid: np.asarray(parameters[vid]) for vid in params if vid in parameters}
    if len(baked) != len(params):
        missing = [vid for vid in params if vid not in parameters]
        raise _FixedPointUnsupported(f"unbaked params: {missing}")

    F = _elems(_shape_of(graph.values[stream_vid]))
    N = cfg.burst
    fx = f"ap_fixed<{cfg.w_bits}, {cfg.i_bits}>"
    acc = f"ap_fixed<{cfg.w_bits + 8}, {cfg.i_bits + 4}>"

    # ---- file-scope prelude: types, LUT, baked weights -------------------
    lut_step = 2.0 * cfg.lut_range / cfg.lut_n
    table = _q(np.tanh(-cfg.lut_range + (np.arange(cfg.lut_n) + 0.5) * lut_step), cfg)
    idx_bits = max(1, (cfg.lut_n - 1).bit_length())
    idx_scale = cfg.lut_n / (2.0 * cfg.lut_range)

    pre: list[str] = [
        f"// Fixed-point II-bound burst-loop: {process.process_id}",
        '#include "ap_fixed.h"',
        '#include "ap_int.h"',
        f"typedef {fx} fx;",
        f"typedef {acc} acc_t;",
        f"static const fx TANH_LUT[{cfg.lut_n}] = " f"{_nested(np.asarray(table))};",
        "static fx tanh_lut(acc_t x) {",
        "#pragma HLS INLINE",
        f"  const fx lo = -{cfg.lut_range}, hi = {cfg.lut_range};",
        "  fx xc = (x < lo) ? lo : ((x > hi) ? hi : (fx)x);",
        f"  const acc_t s = (acc_t)(xc + hi) * (acc_t){idx_scale};",
        "  int idx_i = (int)s;",
        # clamp: at x == +hi, (xc+hi)*idx_scale == lut_n which would wrap the
        # ap_uint index back to 0 (tanh(-hi) instead of tanh(+hi)).
        f"  if (idx_i > {cfg.lut_n - 1}) idx_i = {cfg.lut_n - 1};",
        f"  ap_uint<{idx_bits}> idx = (ap_uint<{idx_bits}>)idx_i;",
        "  return TANH_LUT[idx];",
        "}",
        "static fx sig_lut(acc_t x) {",
        "#pragma HLS INLINE",
        "  const fx half = 0.5;",
        "  return half + half * tanh_lut((acc_t)(half * (fx)x));",
        "}",
        # ReLU needs no table: zero or the fx-truncated input. The oracle
        # mirrors this exactly as max(0, tr_fx(x)).
        "static fx relu_fx(acc_t x) {",
        "#pragma HLS INLINE",
        "  return (x < 0) ? (fx)0 : (fx)x;",
        "}",
    ]
    for custom in cfg.custom_ops.values():
        pre.append(custom.c_body)
    for vid, arr in baked.items():
        shape = _shape_of(graph.values[vid])
        q = _q(arr, cfg).reshape(shape or [1])
        if _is_weight2d(shape):
            pre.append(
                f"static const fx {vid}[{shape[0]}][{shape[1]}] = " f"{_nested(q)};"
            )
        else:
            flat = q.reshape(-1)
            pre.append(
                f"static const fx {vid}[{max(1, flat.size)}] = "
                f"{{{', '.join(_lit(v) for v in flat)}}};"
            )

    # ---- element access helper -------------------------------------------
    def elem(vid: str, k: str) -> str:
        if vid == stream_vid:
            return f"_x[{k}]"
        if vid in state_read:
            return f"{state_read[vid]['state_id']}__rd[{k}]"
        if vid in baked and not _is_weight2d(_shape_of(graph.values[vid])):
            return f"{vid}[{k}]"
        return f"{vid}[{k}]"

    # ---- per-sample body -------------------------------------------------
    body: list[str] = [
        f"  fx _x[{F}];",
        "#pragma HLS ARRAY_PARTITION variable=_x complete",
    ]
    for j in range(F):
        body.append(f"  _x[{j}] = x[t][{j}];")
    # state-read locals
    for b in states:
        sid, elems = b["state_id"], b["elems"]
        body.append(f"  fx {sid}__rd[{elems}];")
        body.append(f"#pragma HLS ARRAY_PARTITION variable={sid}__rd complete")
        body.append(f"  for (int k = 0; k < {elems}; ++k) {{")
        body.append("#pragma HLS UNROLL")
        body.append(f"    {sid}__rd[k] = {sid}__state[k];")
        body.append("  }")
    # internal value declarations
    declared = {stream_vid} | set(state_read)
    for vid in graph.values:
        if vid in declared or vid in baked or vid in state_read:
            continue
        shape = _shape_of(graph.values[vid])
        if _is_weight2d(shape):
            continue
        n = max(1, _elems(shape))
        body.append(f"  fx {vid}[{n}];")
        body.append(f"#pragma HLS ARRAY_PARTITION variable={vid} complete")
        declared.add(vid)

    # op blocks in topological order
    for op_id in _topological_ops(graph):
        op = graph.ops[op_id]
        if op.op_type == "MatMul":
            a, w = op.inputs[0], op.inputs[1]
            out = op.outputs[0]
            wshape = _shape_of(graph.values[w])
            K, M = wshape[0], wshape[1]
            Kp = _pow2(K)
            body.append(f"  {op_id}_mv: for (int m = 0; m < {M}; ++m) {{")
            body.append("#pragma HLS UNROLL")
            body.append(f"    acc_t _p[{Kp}];")
            body.append("#pragma HLS ARRAY_PARTITION variable=_p complete")
            body.append(f"    for (int k = 0; k < {K}; ++k) {{")
            body.append("#pragma HLS UNROLL")
            body.append(f"      _p[k] = (acc_t)({elem(a, 'k')} * {w}[k][m]);")
            body.append("    }")
            if Kp > K:
                body.append(f"    for (int k = {K}; k < {Kp}; ++k) {{")
                body.append("#pragma HLS UNROLL")
                body.append("      _p[k] = (acc_t)0;")
                body.append("    }")
            body.append(f"    for (int s = {Kp // 2}; s > 0; s >>= 1) {{")
            body.append("#pragma HLS UNROLL")
            body.append("      for (int k = 0; k < s; ++k) {")
            body.append("#pragma HLS UNROLL")
            body.append("        _p[k] = _p[k] + _p[k + s];")
            body.append("      }")
            body.append("    }")
            body.append(f"    {out}[m] = (fx)_p[0];")
            body.append("  }")
        elif op.op_type in _BINARY:
            a, b2 = op.inputs[0], op.inputs[1]
            out = op.outputs[0]
            n = max(1, _elems(_shape_of(graph.values[out])))
            body.append(f"  {op_id}_ew: for (int i = 0; i < {n}; ++i) {{")
            body.append("#pragma HLS UNROLL")
            body.append(
                f"    {out}[i] = (fx)((acc_t){elem(a, 'i')} "
                f"{_BINARY[op.op_type]} (acc_t){elem(b2, 'i')});"
            )
            body.append("  }")
        elif op.op_type in _ACT or op.op_type in cfg.custom_ops:
            call = (
                _ACT[op.op_type]
                if op.op_type in _ACT
                else cfg.custom_ops[op.op_type].c_name
            )
            a = op.inputs[0]
            out = op.outputs[0]
            n = max(1, _elems(_shape_of(graph.values[out])))
            body.append(f"  {op_id}_act: for (int i = 0; i < {n}; ++i) {{")
            body.append("#pragma HLS UNROLL")
            body.append(f"    {out}[i] = {call}((acc_t){elem(a, 'i')});")
            body.append("  }")

    # state write-back
    for b in states:
        sid, wv, elems = b["state_id"], b["write"], b["elems"]
        body.append(f"  for (int k = 0; k < {elems}; ++k) {{")
        body.append("#pragma HLS UNROLL")
        body.append(f"    {sid}__state[k] = {wv}[k];")
        body.append("  }")
    body.append(f"  y_out[t] = {out_vid}[0];")

    # ---- assemble DUT ----------------------------------------------------
    state_decls = []
    for b in states:
        sid, elems, init = b["state_id"], b["elems"], b["init"]
        vals = (
            ["0"] * elems
            if all(v == 0.0 for v in init)
            else [_lit(v) for v in (init * elems if len(init) == 1 else init)]
        )
        state_decls.append(
            f"static fx {sid}__state[{elems}] = " f"{{{', '.join(vals)}}};"
        )
        state_decls.append(
            f"#pragma HLS ARRAY_PARTITION variable={sid}__state complete"
        )

    top = f"{process.process_id}_stream"
    dut = "\n".join(
        [
            *pre,
            "",
            f"void {top}(const fx x[{N}][{F}], fx y_out[{N}]) {{",
            "#pragma HLS ARRAY_PARTITION variable=x complete dim=2",
            *[
                f"  {line}" if not line.startswith("#") else line
                for line in state_decls
            ],
            f"  sample_loop: for (int t = 0; t < {N}; ++t) {{",
            f"#pragma HLS PIPELINE II={cfg.target_ii}",
            *body,
            "  }",
            "}",
            "",
        ]
    )

    x_rows, golden_y = _fixedpoint_oracle(process, kernel, baked, golden_trace, cfg)
    tb = _render_tb(x_rows, golden_y, F, N, cfg, top)
    return FixedPointArtifact(dut_hls=dut, testbench_hls=tb, top_name=top)


def _fixedpoint_oracle(process, kernel, baked, golden_trace, cfg):
    """Reference GRU/RNN/LSTM run in the emitter's EXACT ap_fixed semantics.

    Constants (weights, LUT, inputs) are round-to-nearest quantized (as
    baked); runtime arithmetic TRUNCATES toward -inf (ap_fixed default
    AP_TRN) at the modelled precision: products of two fx accumulate in the
    wider acc_t (frac+4 bits) then the reduction output truncates back to
    fx; activations go through the same clamped integer-bitslice LUT. Values
    stay in range so AP_WRAP overflow is a no-op. This makes the testbench
    golden an ORACLE-RELATIVE fixed-point certificate rather than a coarse
    float tolerance.
    """
    graph = kernel.graph
    stream_ins, _params, outs = _step_interface(process, kernel)
    stream_vid, out_vid = stream_ins[0], outs[0]
    states = _state_bindings(process, kernel)
    state_read = {b["read"]: b for b in states}
    state_write = {b["write"]: b["state_id"] for b in states}
    topo = _topological_ops(graph)
    pq = {
        vid: _q(arr, cfg).reshape(_shape_of(graph.values[vid]) or [1])
        for vid, arr in baked.items()
    }

    frac_fx = float(1 << cfg.frac)
    frac_acc = float(1 << (cfg.frac + 4))

    def tr_fx(v):
        return np.floor(np.asarray(v, np.float64) * frac_fx) / frac_fx

    def tr_acc(v):
        return np.floor(np.asarray(v, np.float64) * frac_acc) / frac_acc

    lut_step = 2.0 * cfg.lut_range / cfg.lut_n
    table = _q(np.tanh(-cfg.lut_range + (np.arange(cfg.lut_n) + 0.5) * lut_step), cfg)
    idx_scale = cfg.lut_n / (2.0 * cfg.lut_range)

    def lut_tanh(a):
        xc = np.clip(a, -cfg.lut_range, cfg.lut_range)
        idx = np.floor((xc + cfg.lut_range) * idx_scale).astype(np.int64)
        idx = np.clip(idx, 0, cfg.lut_n - 1)
        return table[idx]

    def lut_sig(a):
        return tr_fx(0.5 + 0.5 * lut_tanh(0.5 * np.asarray(a, np.float64)))

    st = {
        b["state_id"]: (
            np.full(
                b["elems"], b["init"][0] if len(b["init"]) == 1 else 0.0, np.float64
            )
            if all(v == 0.0 for v in b["init"])
            else np.asarray(b["init"], np.float64)
        )
        for b in states
    }

    x_rows, golden_y = [], []
    for step in golden_trace.steps[: cfg.burst]:
        F = _elems(_shape_of(graph.values[stream_vid]))
        xin = _q(
            np.resize(np.asarray(list(step.inputs.values())[0], np.float64).ravel(), F),
            cfg,
        )
        x_rows.append(xin)
        vals = {stream_vid: xin}
        for rvid, b in state_read.items():
            vals[rvid] = st[b["state_id"]].copy()

        def get(vid, _vals):
            # baked bias/vector params are not in vals; fall back to pq flat
            return _vals[vid] if vid in _vals else pq[vid].reshape(-1)

        for op_id in topo:
            op = graph.ops[op_id]
            out = op.outputs[0]
            if op.op_type == "MatMul":
                a, w = get(op.inputs[0], vals), pq[op.inputs[1]]
                res = np.array([np.sum(tr_acc(a * w[:, m])) for m in range(w.shape[1])])
                vals[out] = tr_fx(res)
            elif op.op_type in _BINARY:
                a, b2 = get(op.inputs[0], vals), get(op.inputs[1], vals)
                if op.op_type == "Add":
                    vals[out] = tr_fx(a + b2)
                elif op.op_type == "Sub":
                    vals[out] = tr_fx(a - b2)
                elif op.op_type == "Mul":
                    vals[out] = tr_fx(a * b2)
                else:  # Div
                    vals[out] = tr_fx(a / b2)
            elif op.op_type == "Tanh":
                vals[out] = lut_tanh(get(op.inputs[0], vals))
            elif op.op_type == "Sigmoid":
                vals[out] = lut_sig(get(op.inputs[0], vals))
            elif op.op_type == "ReLU":
                vals[out] = np.maximum(0.0, tr_fx(get(op.inputs[0], vals)))
            elif op.op_type in cfg.custom_ops:
                custom = cfg.custom_ops[op.op_type]
                vals[out] = np.asarray(
                    custom.semantics(get(op.inputs[0], vals)), np.float64
                )
        for wvid, sid in state_write.items():
            st[sid] = vals[wvid].copy()
        golden_y.append(float(np.asarray(vals[out_vid]).ravel()[0]))
    return x_rows, golden_y


def _render_tb(x_rows, golden_y, F, N, cfg, top) -> str:
    xs = ["{" + ", ".join(_lit(v) for v in row) + "}" for row in x_rows]
    ys = [_lit(v) for v in golden_y]
    # Oracle-relative fixed-point certificate: the golden is computed in the
    # emitter's exact ap_fixed semantics, so the gate is a few LSB (residual
    # slack for any un-modelled rounding corner), NOT the old 0.05 float tol.
    tol = 16.0 / cfg.scale
    return "\n".join(
        [
            f"// Asserting TB (oracle-relative fixed-point certificate) for {top}",
            '#include "ap_fixed.h"',
            "#include <cmath>",
            "#include <iostream>",
            f"typedef ap_fixed<{cfg.w_bits}, {cfg.i_bits}> fx;",
            f"extern void {top}(const fx x[{N}][{F}], fx y[{N}]);",
            "int main() {",
            f"  static const fx x[{N}][{F}] = {{",
            "  " + ",\n  ".join(xs),
            "  };",
            f"  static const double golden[{N}] = {{{', '.join(ys)}}};",
            f"  fx y[{N}];",
            f"  {top}(x, y);",
            "  int errors = 0;",
            f"  for (int t = 0; t < {N}; ++t) {{",
            f"    if (std::fabs((double)y[t] - golden[t]) > {tol:g}) {{",
            '      std::cerr << "MISMATCH t=" << t << " got " << (double)y[t]',
            '                << " want " << golden[t] << std::endl;',
            "      ++errors;",
            "    }",
            "  }",
            '  std::cout << "FX burst TB complete, errors=" << errors' " << std::endl;",
            "  return errors == 0 ? 0 : 1;",
            "}",
            "",
        ]
    )


def write_fixedpoint_burst_bundle(
    process: Process,
    golden_trace: GoldenTrace,
    output_dir: str | PathLike[str],
    parameters: dict[str, object],
    *,
    stem: str | None = None,
    config: FixedPointConfig | None = None,
) -> dict[str, object]:
    cfg = config or FixedPointConfig()
    art = render_fixedpoint_burst_artifact(process, golden_trace, parameters, cfg)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    art_stem = stem or process.process_id
    (out / f"{art_stem}.cpp").write_text(art.dut_hls, encoding="utf-8")
    (out / f"{art_stem}_tb.cpp").write_text(art.testbench_hls, encoding="utf-8")
    return {
        "top": art.top_name,
        "stem": art_stem,
        "dut": out / f"{art_stem}.cpp",
        "tb": out / f"{art_stem}_tb.cpp",
    }
