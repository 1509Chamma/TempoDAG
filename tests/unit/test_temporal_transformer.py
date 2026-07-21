"""Contract tests for the fixed-point transformer (PatchTST) block emitter."""

from __future__ import annotations

import numpy as np

from tempo_dag.codegen.hls.temporal_transformer_generator import (
    TransformerConfig,
    _block_oracle,
    _Luts,
    demo_block_weights,
    render_transformer_artifact,
    write_transformer_bundle,
)


def _float_block(X, W, D):
    def ln(z):
        m = z.mean(-1, keepdims=True)
        return (z - m) / np.sqrt(z.var(-1, keepdims=True) + 1e-5)

    def gelu(x):
        return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))

    def softmax(s):
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(-1, keepdims=True)

    h = ln(X)
    Q, K, V = h @ W["Wq"], h @ W["Wk"], h @ W["Wv"]
    A = softmax((Q @ K.T) / np.sqrt(D))
    x1 = X + (A @ V) @ W["Wo"]
    h2 = ln(x1)
    return x1 + gelu(h2 @ W["W1"]) @ W["W2"]


def test_transformer_emits_block_contract():
    cfg = TransformerConfig()
    dut, tb, top, X, golden = render_transformer_artifact(cfg, demo_block_weights(cfg))
    assert top == "patchtst_block"
    # wider datapath than the recurrent archs (headroom for scores/FFN)
    assert "typedef ap_fixed<24, 8> fx;" in dut
    # the three new LUT primitives + reuse of the integer-bitslice machinery
    for tab in ("EXP_TAB", "GELU_TAB", "RSQ_TAB"):
        assert f"static const fx {tab}[1024]" in dut
    assert "EXP_lut" in dut and "GELU_lut" in dut and "RSQ_lut" in dut
    assert "std::exp" not in dut and "std::tanh" not in dut
    # attention + FFN structure
    assert "mm_scores(" in dut and "mm_ctx(" in dut and "mm_ff1(" in dut
    assert "smax:" in dut and "ln1:" in dut and "ln2:" in dut
    # DSP-bound matmul: pipeline over outputs (i,m), tree-reduce only the
    # K-wide reduction (tractable) - NOT unrolling all i*m*k (explodes bind)
    assert "p[k] = (acc_t)(A[i][k] * W[k][m]);" in dut
    assert "p[kk] = p[kk] + p[kk + s];" in dut
    assert "#pragma HLS PIPELINE II=1" in dut
    # brace balance
    assert dut.count("{") == dut.count("}")


def test_transformer_oracle_tracks_float():
    cfg = TransformerConfig()
    W = demo_block_weights(cfg)
    luts = _Luts(cfg)
    rng = np.random.default_rng(cfg.seed + 1)
    X = rng.standard_normal((cfg.t_patches, cfg.d_model)) * 0.5
    oracle = _block_oracle(X, W, cfg, luts)
    ref = _float_block(X, W, cfg.d_model)
    # LUT softmax/gelu/rsqrt + fixed point: bounded error (NB24 ~0.035)
    assert np.max(np.abs(oracle - ref)) < 0.08


def test_transformer_testbench_asserts_golden():
    cfg = TransformerConfig()
    _dut, tb, _top, _X, _g = render_transformer_artifact(cfg, demo_block_weights(cfg))
    assert "static const double golden" in tb
    assert "++errors" in tb and "return errors == 0 ? 0 : 1;" in tb


def test_write_transformer_bundle_writes_files(tmp_path):
    info = write_transformer_bundle(tmp_path, stem="patchtst")
    assert info["top"] == "patchtst_block"
    dut, tb = tmp_path / "patchtst.cpp", tmp_path / "patchtst_tb.cpp"
    assert dut.exists() and tb.exists()
    assert "patchtst_block" in dut.read_text()
    assert "PatchTST block TB complete" in tb.read_text()
