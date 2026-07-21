"""Streaming-inference benchmark: architectures x backends, per-sample.

Protocol: batch=1, state carried, warmup then timed steps, median/p95/p99
ns per sample. Parity checked against the numpy reference where available.
Backends that cannot run record `unsupported` with a reason.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO / "research"),):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS = HERE / "results"
WARMUP, TIMED = 200, 2000
SEED = 0
H = 16          # hidden size (realistic-small edge deployment)
FEATURES = 4    # input features per sample


# ------------------------- datasets -------------------------
def synth_finance(n: int) -> np.ndarray:
    """Seeded random-walk price stream -> per-tick feature vector."""
    rng = np.random.default_rng(SEED)
    price = np.cumsum(0.01 * rng.standard_normal(n + 64)) + 100.0
    ret = np.diff(price)
    feats = np.stack([
        ret[63:],                                    # last return
        np.convolve(ret, np.ones(8) / 8, "valid")[-n:],   # 8-tick mean
        np.convolve(ret, np.ones(32) / 32, "valid")[-n:], # 32-tick mean
        np.abs(ret[63:]),                            # abs move
    ], axis=1).astype(np.float32)
    return feats[:n]


DATASETS = {"synth_finance": synth_finance}


# ------------------------- models (numpy reference) -------------------------
def make_weights(arch: str) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)

    def m(*shape, scale=0.4):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    w = {"Wo": m(H, 1), "bo": m(1)}
    if arch in ("rnn", "gru", "lstm"):
        gates = {"rnn": 1, "gru": 3, "lstm": 4}[arch]
        w |= {"Wx": m(gates * H, FEATURES), "Wh": m(gates * H, H),
              "b": m(gates * H)}
        # stabilize recurrent block (research finding: spectral norm matters)
        Wh = w["Wh"].reshape(gates, H, H)
        for g in range(gates):
            Wh[g] /= max(1e-6, np.linalg.norm(Wh[g], 2)) / 0.9
        w["Wh"] = Wh.reshape(gates * H, H)
    elif arch == "transformer":
        w |= {"Wq": m(H, FEATURES), "Wk": m(H, FEATURES), "Wv": m(H, FEATURES),
              "Wff": m(H, H)}
    return w


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyModel:
    """Streaming reference implementations (float32, per-sample)."""

    def __init__(self, arch: str, w: dict[str, np.ndarray]):
        self.arch, self.w = arch, w
        self.reset()

    def reset(self):
        self.h = np.zeros(H, np.float32)
        self.c = np.zeros(H, np.float32)
        self.ema = np.float32(0.0)
        self.var = np.float32(1.0)
        self.kv: list[np.ndarray] = []

    def step(self, x: np.ndarray) -> float:
        w = self.w
        if self.arch == "statistical":
            r = x[0]
            self.ema = 0.94 * self.ema + 0.06 * r
            self.var = 0.94 * self.var + 0.06 * (r - self.ema) ** 2
            return float((r - self.ema) / np.sqrt(self.var + 1e-8))
        if self.arch == "rnn":
            self.h = np.tanh(w["Wx"] @ x + w["Wh"] @ self.h + w["b"])
        elif self.arch == "gru":
            z_ = w["Wx"] @ x + w["Wh"] @ self.h + w["b"]
            r = sigmoid(z_[:H])
            z = sigmoid(z_[H:2 * H])
            n = np.tanh(z_[2 * H:] * r)  # simplified coupled-gate GRU
            self.h = (1 - z) * n + z * self.h
        elif self.arch == "lstm":
            z_ = w["Wx"] @ x + w["Wh"] @ self.h + w["b"]
            i, f, g, o = (sigmoid(z_[:H]), sigmoid(z_[H:2 * H]),
                          np.tanh(z_[2 * H:3 * H]), sigmoid(z_[3 * H:]))
            self.c = f * self.c + i * g
            self.h = o * np.tanh(self.c)
        elif self.arch == "transformer":
            q = w["Wq"] @ x
            self.kv.append(np.stack([w["Wk"] @ x, w["Wv"] @ x]))
            if len(self.kv) > 32:          # causal window / KV cache cap
                self.kv.pop(0)
            ks = np.stack([kv[0] for kv in self.kv])
            vs = np.stack([kv[1] for kv in self.kv])
            att = ks @ q / np.sqrt(H)
            att = np.exp(att - att.max())
            att /= att.sum()
            self.h = np.tanh(w["Wff"] @ (att @ vs))
        return float((w["Wo"].T @ self.h + w["bo"])[0])


# ------------------------- torch backends -------------------------
def build_torch(arch, w, device, compiled):
    try:
        import torch
    except ImportError:
        return None, "torch not installed"
    if device == "cuda" and not torch.cuda.is_available():
        return None, "no CUDA device"
    tw = {k: torch.tensor(v, device=device) for k, v in w.items()}
    ref = NumpyModel(arch, w)

    class T:
        def __init__(self):
            self.reset()

        def reset(self):
            self.h = torch.zeros(H, device=device)
            self.c = torch.zeros(H, device=device)
            self.ema = torch.zeros((), device=device)
            self.var = torch.ones((), device=device)
            self.kv = []

        def step_fn(self, x, h, c):
            if arch == "statistical":
                ema = 0.94 * self.ema + 0.06 * x[0]
                var = 0.94 * self.var + 0.06 * (x[0] - ema) ** 2
                self.ema, self.var = ema, var
                return (x[0] - ema) / torch.sqrt(var + 1e-8), h, c
            z_ = tw["Wx"] @ x + tw["Wh"] @ h + tw["b"] \
                if arch in ("rnn", "gru", "lstm") else None
            if arch == "rnn":
                h = torch.tanh(z_)
            elif arch == "gru":
                r, z = torch.sigmoid(z_[:H]), torch.sigmoid(z_[H:2 * H])
                n = torch.tanh(z_[2 * H:] * r)
                h = (1 - z) * n + z * h
            elif arch == "lstm":
                i, f = torch.sigmoid(z_[:H]), torch.sigmoid(z_[H:2 * H])
                g, o = torch.tanh(z_[2 * H:3 * H]), torch.sigmoid(z_[3 * H:])
                c = f * c + i * g
                h = o * torch.tanh(c)
            elif arch == "transformer":
                q = tw["Wq"] @ x
                self.kv.append(torch.stack([tw["Wk"] @ x, tw["Wv"] @ x]))
                if len(self.kv) > 32:
                    self.kv.pop(0)
                ks = torch.stack([kv[0] for kv in self.kv])
                vs = torch.stack([kv[1] for kv in self.kv])
                att = torch.softmax(ks @ q / (H ** 0.5), dim=0)
                h = torch.tanh(tw["Wff"] @ (att @ vs))
            return tw["Wo"].T @ h + tw["bo"], h, c

    model = T()
    fn = model.step_fn
    if compiled:
        try:
            fn = torch.compile(model.step_fn, dynamic=False)
        except Exception as exc:  # e.g. no C++ toolchain on Windows
            return None, f"torch.compile unavailable: {type(exc).__name__}"
    del ref
    return (model, fn, tw), None


# ------------------------- timing protocol -------------------------
def run_backend(arch, backend, data):
    w = make_weights(arch)
    if backend == "numpy":
        model = NumpyModel(arch, w)
        ref_out = []
        times = []
        for i, x in enumerate(data[:WARMUP + TIMED]):
            t0 = time.perf_counter_ns()
            y = model.step(x)
            times.append(time.perf_counter_ns() - t0)
            if i >= WARMUP:
                ref_out.append(y)
        return times[WARMUP:], ref_out, None
    if backend.startswith("torch"):
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            return None, None, "torch not installed"
        import torch
        device = "cuda" if backend.endswith("cuda") else "cpu"
        compiled = "compile" in backend
        built, reason = build_torch(arch, w, device, compiled)
        if built is None:
            return None, None, reason
        model, fn, _ = built
        try:  # compile is lazy: probe one call so failures become 'unsupported'
            probe = torch.tensor(data[0], device=device)
            fn(probe, model.h, model.c)
            model.reset()
        except Exception as exc:
            return None, None, (f"{type(exc).__name__}"
                                " (torch.compile needs MSVC cl.exe on Windows)")
        times, outs = [], []
        with torch.no_grad():
            for i, x in enumerate(data[:WARMUP + TIMED]):
                xt = torch.tensor(x, device=device)
                t0 = time.perf_counter_ns()
                y, model.h, model.c = fn(xt, model.h, model.c)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter_ns() - t0)
                if i >= WARMUP:
                    outs.append(float(y))
        return times[WARMUP:], outs, None
    if backend in ("hls4ml", "tempodag"):
        reasons = {
            "hls4ml": "pending: no stateful streaming cell in mature flow "
                      "(library-positioning.md C1); planned comparison",
            "tempodag": "pending: wiring registry covers {RollingMean,Conv1D,"
                        "Add}; extend to this arch then run the Vitis ladder",
        }
        return None, None, reasons[backend]
    return None, None, f"unknown backend {backend}"


def provenance():
    from lab.provenance import provenance as prov
    return prov(seed=SEED, config={"H": H, "features": FEATURES,
                                   "warmup": WARMUP, "timed": TIMED})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="all")
    ap.add_argument("--dataset", default="synth_finance")
    ap.add_argument("--backends", default="numpy,torch_cpu,torch_compile_cpu,"
                                          "torch_cuda,hls4ml,tempodag")
    args = ap.parse_args()

    archs = (["statistical", "rnn", "gru", "lstm", "transformer"]
             if args.arch == "all" else [args.arch])
    data = DATASETS[args.dataset](WARMUP + TIMED)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "results.jsonl"
    prov = provenance()

    print(f"{'arch':>12} {'backend':>18} {'median_us':>10} {'p99_us':>8} "
          f"{'ksamp/s':>8} {'parity':>7}  note")
    for arch in archs:
        ref_outputs = None
        for backend in args.backends.split(","):
            times, outs, reason = run_backend(arch, backend, data)
            if times is None:
                print(f"{arch:>12} {backend:>18} {'-':>10} {'-':>8} {'-':>8} "
                      f"{'-':>7}  unsupported: {reason}")
                row = {"arch": arch, "backend": backend, "dataset": args.dataset,
                       "status": "unsupported", "reason": reason}
            else:
                if backend == "numpy":
                    ref_outputs = outs
                med = statistics.median(times)
                p99 = statistics.quantiles(times, n=100)[98]
                parity = "-"
                if ref_outputs is not None and outs is not None \
                        and backend != "numpy":
                    err = max(abs(a - b) for a, b
                              in zip(outs, ref_outputs, strict=False))
                    parity = "OK" if err < 1e-3 else f"{err:.1e}"
                print(f"{arch:>12} {backend:>18} {med / 1000:>10.2f} "
                      f"{p99 / 1000:>8.1f} {1e6 / med:>8.1f} {parity:>7}")
                row = {"arch": arch, "backend": backend, "dataset": args.dataset,
                       "status": "ok", "median_ns": med, "p99_ns": p99,
                       "samples_per_sec": 1e9 / med, "parity": parity}
            row["provenance"] = prov
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
    print(f"\nplatform: {platform.processor() or platform.machine()}")
    print(f"rows appended -> {out}")


if __name__ == "__main__":
    main()
