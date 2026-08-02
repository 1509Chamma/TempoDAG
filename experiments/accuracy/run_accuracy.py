"""Accuracy retention across real tasks: train -> deploy -> measure.

The question a deployment claim must answer: how much task accuracy does the
Q6.12 fixed-point deployment cost, on real workloads? This harness trains
small streaming models on three tasks, deploys each through the exact
compiler path (weights -> temporal process -> parity proof -> fixed-point
oracle), and reports float-vs-deployed metrics side by side.

Why the oracle stands in for hardware: the fixed-point oracle implements the
emitter's exact ap_fixed semantics, and the cost-model validation campaign
verified oracle-vs-RTL agreement by C/RTL co-simulation across the design
suite. Oracle accuracy therefore *is* deployed accuracy, and this harness
needs no FPGA toolchain.

Tasks:
  mackey   Mackey-Glass chaotic forecasting (synthetic, generated in-place)
  narma    NARMA-10 system identification (the reservoir-computing standard)
  ecg      ECG5000 (real heartbeats, UCR archive; binary normal-vs-abnormal;
           downloaded on first use, cached under data/)

Models (chosen to span the latency classes from the cost-model campaign):
  gru      PyTorch-exact GRU cell         (matmul-in-loop class, 60 ns/sample)
  mingru   minGRU (Feng et al. 2024)      (elementwise class, 20 ns/sample)
  diag     diagonal-linear SSM            (elementwise class, 20 ns/sample)

Usage:
  python experiments/accuracy/run_accuracy.py                 # all tasks x models
  python experiments/accuracy/run_accuracy.py --tasks mackey  # subset
  python experiments/accuracy/run_accuracy.py --report        # table from results
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tempo_dag.codegen.hls.temporal_fixedpoint_generator import (  # noqa: E402
    FixedPointConfig,
    _fixedpoint_oracle,
)
from tempo_dag.ir.graph import Graph  # noqa: E402
from tempo_dag.ir.value import Value, ValueType  # noqa: E402
from tempo_dag.ir_temporal import (  # noqa: E402
    Edge0,
    EdgeDelta,
    Kernel,
    Process,
    StateKind,
    StateSpec,
)
from tempo_dag.ops.builtins import Add, MatMul, Mul, Sigmoid, Sub, Tanh  # noqa: E402
from tempo_dag.verification.golden_trace import GoldenTraceRecorder  # noqa: E402
from tempo_dag.verification.temporal_parity import (  # noqa: E402
    TemporalExecutionTrace,
    TemporalTraceStep,
)

SEED = 20260801
H = 16
DATA = HERE / "data"
RESULTS = HERE / "results"
RESULTS_FILE = RESULTS / "accuracy.jsonl"
ECG_URL = "https://timeseriesclassification.com/aeon-toolkit/ECG5000.zip"

torch.manual_seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------------------------------
# Tasks. Each returns (x_train, y_train, x_test, y_test, kind, reset_len).
# reset_len > 0 means state resets every reset_len steps (per-sequence tasks).
# --------------------------------------------------------------------------

def task_mackey():
    """Mackey-Glass (tau=17) one-step-ahead forecasting from 4 lags."""
    n, tau = 3100, 17
    x = np.zeros(n + tau)
    x[:tau] = 1.2
    for t in range(tau, n + tau - 1):
        x[t + 1] = x[t] + 0.2 * x[t - tau] / (1 + x[t - tau] ** 10) - 0.1 * x[t]
    s = x[tau + 100:].astype(np.float32)          # drop transient
    hz = 12                                       # forecast horizon: the MG
    feats = np.stack([s[3 - k:len(s) - 1 - k] for k in range(4)], axis=1)
    targets = s[3 + hz:]                          # standard 12-ahead setting
    n = min(len(feats), len(targets))
    feats, targets = feats[:n], targets[:n]
    return feats[:2000], targets[:2000], feats[2000:2500], targets[2000:2500], \
        "regression", 0


def task_narma():
    """NARMA-10 (Atiya & Parlos): the standard nonlinear system-id benchmark."""
    rng = np.random.default_rng(SEED)
    n = 2700
    u = rng.uniform(0, 0.5, n).astype(np.float32)
    y = np.zeros(n, np.float32)
    for t in range(9, n - 1):
        y[t + 1] = (0.3 * y[t] + 0.05 * y[t] * y[t - 9:t + 1].sum()
                    + 1.5 * u[t - 9] * u[t] + 0.1)
    feats = np.stack([u[3 - k:n - 1 - k] for k in range(4)], axis=1)
    targets = y[4:]
    keep = slice(100, 2600)                        # drop warmup
    feats, targets = feats[keep], targets[keep]
    return feats[:2000], targets[:2000], feats[2000:], targets[2000:], \
        "regression", 0


def task_ecg():
    """ECG5000 (UCR archive): 140-sample heartbeats, binary normal/abnormal."""
    DATA.mkdir(exist_ok=True)
    cache = DATA / "ECG5000.npz"
    if not cache.exists():
        print("downloading ECG5000 ...")
        raw = urllib.request.urlopen(ECG_URL, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            def load(name):
                rows = [line.split() for line in
                        z.read(name).decode().strip().splitlines()]
                a = np.array(rows, np.float32)
                return a[:, 1:], (a[:, 0] == 1).astype(np.float32)
            xtr, ytr = load("ECG5000_TRAIN.txt")
            xte, yte = load("ECG5000_TEST.txt")
        np.savez(cache, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    d = np.load(cache)
    xtr, ytr = d["xtr"], d["ytr"]
    # The UCR test file is sorted by class - slicing without shuffling
    # yields a single-class test set (caught by the baseline column).
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(d["xte"]))[:1000]
    xte, yte = d["xte"][order], d["yte"][order]
    mu, sd = xtr.mean(), xtr.std()
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    return xtr[..., None], ytr, xte[..., None], yte, "classification", 140


TASKS = {"mackey": task_mackey, "narma": task_narma, "ecg": task_ecg}


# --------------------------------------------------------------------------
# Models: torch training + IR builder from the trained weights.
# All cells are exactly the forms the compiler emits (campaign builders).
# --------------------------------------------------------------------------

class TorchMinGRU(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.wz = nn.Linear(f, H, bias=False)
        self.wh = nn.Linear(f, H, bias=False)
        self.head = nn.Linear(H, 1)

    def forward(self, x):                          # x: [B, T, F]
        h = torch.zeros(x.shape[0], H)
        ys = []
        for t in range(x.shape[1]):
            z = torch.sigmoid(self.wz(x[:, t]))
            h = (1 - z) * h + z * self.wh(x[:, t])
            ys.append(self.head(h))
        return torch.stack(ys, 1)


class TorchDiag(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.a_raw = nn.Parameter(torch.randn(H) * 0.5 + 1.0)
        self.b = nn.Linear(f, H, bias=False)
        self.head = nn.Linear(H, 1)

    def a(self):
        return torch.sigmoid(self.a_raw)           # stable decay in (0, 1)

    def forward(self, x):
        h = torch.zeros(x.shape[0], H)
        ys = []
        for t in range(x.shape[1]):
            h = self.a() * h + self.b(x[:, t])
            ys.append(self.head(h))
        return torch.stack(ys, 1)


class TorchGRU(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.gru = nn.GRU(f, H, batch_first=True)
        self.head = nn.Linear(H, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out)


def val(vid, shape, layout=None):
    return Value(value_id=vid, vtype=ValueType.TENSOR, dtype="float32",
                 shape=shape, axes=[f"a{i}" for i in range(len(shape))],
                 layout=layout)


def _proc(pid, values, ops, gin, gout, f):
    g = Graph(values=values, ops=ops, graph_inputs=gin, graph_outputs=gout)
    p = Process(process_id=pid, kernels={"k": Kernel("k", graph=g)},
                states={"h": StateSpec("h", StateKind.HIDDEN, "float32", (H,))},
                edge0=[Edge0("h", "k", value_id="h_prev")],
                edge_delta=[EdgeDelta("k", "h", lag_cycles=1,
                                      value_id="h_next")])
    p.validate()
    return p


def build_gru_proc(model, f):
    """PyTorch-exact GRU: reset gate on the hidden-side candidate term only."""
    sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    Wi, Wh = sd["gru.weight_ih_l0"], sd["gru.weight_hh_l0"]
    bi, bh = sd["gru.bias_ih_l0"], sd["gru.bias_hh_l0"]
    g = {}
    for i, name in enumerate(("r", "z", "n")):     # torch gate order
        sl = slice(i * H, (i + 1) * H)
        g[name] = dict(Wx=Wi[sl], Wh=Wh[sl], bx=bi[sl], bh=bh[sl])
    values = {"x": val("x", [1, f]), "h_prev": val("h_prev", [1, H]),
              "h_next": val("h_next", [1, H]),
              "ones": val("ones", [1, H], layout="parameter"),
              "y0": val("y0", [1, 1]), "y": val("y", [1, 1]),
              "WoT": val("WoT", [H, 1], layout="parameter"),
              "bo": val("bo", [1, 1], layout="parameter")}
    ops = {}
    params = {"ones": np.ones((1, H), np.float32),
              "WoT": sd["head.weight"].T.copy(),
              "bo": sd["head.bias"][None, :].copy()}
    for gn in ("r", "z"):
        for vid in (f"xg_{gn}", f"hg_{gn}", f"sg_{gn}", f"pre_{gn}", f"{gn}g"):
            values[vid] = val(vid, [1, H])
        values[f"WxT_{gn}"] = val(f"WxT_{gn}", [f, H], layout="parameter")
        values[f"WhT_{gn}"] = val(f"WhT_{gn}", [H, H], layout="parameter")
        values[f"b_{gn}"] = val(f"b_{gn}", [1, H], layout="parameter")
        ops[f"mmx_{gn}"] = MatMul(f"mmx_{gn}", inputs=["x", f"WxT_{gn}"],
                                  outputs=[f"xg_{gn}"])
        ops[f"mmh_{gn}"] = MatMul(f"mmh_{gn}", inputs=["h_prev", f"WhT_{gn}"],
                                  outputs=[f"hg_{gn}"])
        ops[f"gs_{gn}"] = Add(f"gs_{gn}", inputs=[f"xg_{gn}", f"hg_{gn}"],
                              outputs=[f"sg_{gn}"])
        ops[f"gb_{gn}"] = Add(f"gb_{gn}", inputs=[f"sg_{gn}", f"b_{gn}"],
                              outputs=[f"pre_{gn}"])
        ops[f"sig_{gn}"] = Sigmoid(f"sig_{gn}", inputs=[f"pre_{gn}"],
                                   outputs=[f"{gn}g"])
        params[f"WxT_{gn}"] = g[gn]["Wx"].T.copy()
        params[f"WhT_{gn}"] = g[gn]["Wh"].T.copy()
        params[f"b_{gn}"] = (g[gn]["bx"] + g[gn]["bh"])[None, :].copy()
    for vid in ("xg_n", "xgb_n", "hg_n", "hgb_n", "nr", "pre_n", "n",
                "omz", "part_a", "part_b"):
        values[vid] = val(vid, [1, H])
    for vid, shape in (("WxT_n", [f, H]), ("WhT_n", [H, H]),
                       ("bx_n", [1, H]), ("bh_n", [1, H])):
        values[vid] = val(vid, shape, layout="parameter")
    ops["mmx_n"] = MatMul("mmx_n", inputs=["x", "WxT_n"], outputs=["xg_n"])
    ops["bx_add"] = Add("bx_add", inputs=["xg_n", "bx_n"], outputs=["xgb_n"])
    ops["mmh_n"] = MatMul("mmh_n", inputs=["h_prev", "WhT_n"], outputs=["hg_n"])
    ops["bh_add"] = Add("bh_add", inputs=["hg_n", "bh_n"], outputs=["hgb_n"])
    ops["mul_nr"] = Mul("mul_nr", inputs=["rg", "hgb_n"], outputs=["nr"])
    ops["pre_add"] = Add("pre_add", inputs=["xgb_n", "nr"], outputs=["pre_n"])
    ops["tanh_n"] = Tanh("tanh_n", inputs=["pre_n"], outputs=["n"])
    ops["one_minus"] = Sub("one_minus", inputs=["ones", "zg"], outputs=["omz"])
    ops["mul_a"] = Mul("mul_a", inputs=["omz", "n"], outputs=["part_a"])
    ops["mul_b"] = Mul("mul_b", inputs=["zg", "h_prev"], outputs=["part_b"])
    ops["h_upd"] = Add("h_upd", inputs=["part_a", "part_b"], outputs=["h_next"])
    ops["mm_out"] = MatMul("mm_out", inputs=["h_next", "WoT"], outputs=["y0"])
    ops["add_out"] = Add("add_out", inputs=["y0", "bo"], outputs=["y"])
    params["WxT_n"] = g["n"]["Wx"].T.copy()
    params["WhT_n"] = g["n"]["Wh"].T.copy()
    params["bx_n"] = g["n"]["bx"][None, :].copy()
    params["bh_n"] = g["n"]["bh"][None, :].copy()
    p = _proc("acc_gru", values, ops,
              ["x", "h_prev"] + [k for k, v in values.items()
                                 if v.layout == "parameter"],
              ["y", "h_next"], f)
    return p, params


def build_mingru_proc(model, f):
    sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    values = {"x": val("x", [1, f]), "h_prev": val("h_prev", [1, H]),
              "h_next": val("h_next", [1, H]),
              "Wz": val("Wz", [f, H], layout="parameter"),
              "Wh": val("Wh", [f, H], layout="parameter"),
              "ones": val("ones", [1, H], layout="parameter"),
              "y0": val("y0", [1, 1]), "y": val("y", [1, 1]),
              "WoT": val("WoT", [H, 1], layout="parameter"),
              "bo": val("bo", [1, 1], layout="parameter")}
    for vid in ("zx", "zg", "ht", "omz", "pa", "pb"):
        values[vid] = val(vid, [1, H])
    ops = {
        "mm_z": MatMul("mm_z", inputs=["x", "Wz"], outputs=["zx"]),
        "sig_z": Sigmoid("sig_z", inputs=["zx"], outputs=["zg"]),
        "mm_h": MatMul("mm_h", inputs=["x", "Wh"], outputs=["ht"]),
        "omz": Sub("omz", inputs=["ones", "zg"], outputs=["omz"]),
        "pa": Mul("pa", inputs=["omz", "h_prev"], outputs=["pa"]),
        "pb": Mul("pb", inputs=["zg", "ht"], outputs=["pb"]),
        "h_upd": Add("h_upd", inputs=["pa", "pb"], outputs=["h_next"]),
        "mm_out": MatMul("mm_out", inputs=["h_next", "WoT"], outputs=["y0"]),
        "add_out": Add("add_out", inputs=["y0", "bo"], outputs=["y"]),
    }
    params = {"ones": np.ones((1, H), np.float32),
              "Wz": sd["wz.weight"].T.copy(), "Wh": sd["wh.weight"].T.copy(),
              "WoT": sd["head.weight"].T.copy(),
              "bo": sd["head.bias"][None, :].copy()}
    p = _proc("acc_mingru", values, ops,
              ["x", "h_prev"] + [k for k, v in values.items()
                                 if v.layout == "parameter"],
              ["y", "h_next"], f)
    return p, params


def build_diag_proc(model, f):
    sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    a = 1.0 / (1.0 + np.exp(-sd["a_raw"]))
    values = {"x": val("x", [1, f]), "h_prev": val("h_prev", [1, H]),
              "h_next": val("h_next", [1, H]),
              "a_diag": val("a_diag", [1, H], layout="parameter"),
              "B": val("B", [f, H], layout="parameter"),
              "bx": val("bx", [1, H]), "ah": val("ah", [1, H]),
              "y0": val("y0", [1, 1]), "y": val("y", [1, 1]),
              "WoT": val("WoT", [H, 1], layout="parameter"),
              "bo": val("bo", [1, 1], layout="parameter")}
    ops = {
        "mm_bx": MatMul("mm_bx", inputs=["x", "B"], outputs=["bx"]),
        "mul_ah": Mul("mul_ah", inputs=["a_diag", "h_prev"], outputs=["ah"]),
        "h_upd": Add("h_upd", inputs=["ah", "bx"], outputs=["h_next"]),
        "mm_out": MatMul("mm_out", inputs=["h_next", "WoT"], outputs=["y0"]),
        "add_out": Add("add_out", inputs=["y0", "bo"], outputs=["y"]),
    }
    params = {"a_diag": a[None, :].astype(np.float32),
              "B": sd["b.weight"].T.copy(),
              "WoT": sd["head.weight"].T.copy(),
              "bo": sd["head.bias"][None, :].copy()}
    p = _proc("acc_diag", values, ops,
              ["x", "h_prev"] + [k for k, v in values.items()
                                 if v.layout == "parameter"],
              ["y", "h_next"], f)
    return p, params


MODELS = {
    "gru": (TorchGRU, build_gru_proc),
    "mingru": (TorchMinGRU, build_mingru_proc),
    "diag": (TorchDiag, build_diag_proc),
}


# --------------------------------------------------------------------------
# Training, float parity, fixed-point evaluation
# --------------------------------------------------------------------------

def train(model_cls, x, y, kind, reset_len, f, epochs=400):
    torch.manual_seed(SEED)
    model = model_cls(f)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if reset_len:                                  # per-sequence task
        xb = torch.from_numpy(x)
        yb = torch.from_numpy(y)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            opt.zero_grad()
            out = model(xb)[:, -1, 0]              # logit at final step
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            sched.step()
    else:                                          # one long stream
        xb = torch.from_numpy(x[None])
        yb = torch.from_numpy(y[None, :, None])
        for _ in range(epochs):
            opt.zero_grad()
            loss = ((model(xb) - yb) ** 2).mean()
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return model


SEM = {"MatMul": lambda a, b: a @ b, "Add": lambda a, b: a + b,
       "Sub": lambda a, b: a - b, "Mul": lambda a, b: a * b,
       "Sigmoid": lambda a: 1.0 / (1.0 + np.exp(-a)), "Tanh": np.tanh}


def run_float(process, params, xs):
    """Float32 reference over the IR graph (the parity oracle)."""
    g = process.kernels["k"].graph
    h = np.zeros((1, H), np.float32)
    ys = []
    for x in xs:
        env = {"x": x[None, :].astype(np.float32), "h_prev": h,
               **{k: np.asarray(v, np.float32) for k, v in params.items()}}
        pending = dict(g.ops)
        while pending:
            for name, op in list(pending.items()):
                if all(i in env for i in op.inputs):
                    env[op.outputs[0]] = SEM[op.op_type](*[env[i]
                                                           for i in op.inputs])
                    del pending[name]
        h = env["h_next"]
        ys.append(float(env["y"][0, 0]))
    return np.array(ys)


def run_fixed(process, params, xs, cfg):
    """Deployed (Q-format) outputs via the emitter's exact oracle."""
    steps = [TemporalTraceStep(timestep=i, inputs={"x": xs[i].astype(np.float64)},
                               outputs={"y": np.array([0.0])},
                               state={"h": np.zeros(H)})
             for i in range(len(xs))]
    trace = GoldenTraceRecorder().record(
        TemporalExecutionTrace(tuple(steps)), metadata={"case": "acc"})
    kernel = process.kernels["k"]
    _, golden_y = _fixedpoint_oracle(process, kernel, params, trace, cfg)
    return np.array(golden_y)


def nrmse(pred, target):
    return float(np.sqrt(np.mean((pred - target) ** 2)) / (target.std() + 1e-12))


def evaluate(task_name, model_name, only_report=False):
    x_tr, y_tr, x_te, y_te, kind, reset_len = TASKS[task_name]()
    f = x_tr.shape[-1]
    model_cls, builder = MODELS[model_name]
    t0 = time.time()
    model = train(model_cls, x_tr, y_tr, kind, reset_len, f)
    process, params = builder(model, f)

    if reset_len:                                  # per-sequence evaluation
        with torch.no_grad():
            torch_logits = model(torch.from_numpy(x_te))[:, -1, 0].numpy()
        fl_logits, fx_logits = [], []
        cfg = FixedPointConfig(burst=reset_len)
        for seq in x_te:
            fl_logits.append(run_float(process, params, seq)[-1])
            fx_logits.append(run_fixed(process, params, seq, cfg)[-1])
        fl_logits, fx_logits = np.array(fl_logits), np.array(fx_logits)
        parity = float(np.abs(torch_logits - fl_logits).max())
        base = float(max(y_te.mean(), 1 - y_te.mean()))
        m_fl = float(((fl_logits > 0) == (y_te > 0.5)).mean())
        m_fx = float(((fx_logits > 0) == (y_te > 0.5)).mean())
        metric, delta = "accuracy", m_fx - m_fl
    else:
        with torch.no_grad():
            torch_y = model(torch.from_numpy(x_te[None]))[0, :, 0].numpy()
        fl = run_float(process, params, x_te)
        cfg = FixedPointConfig(burst=len(x_te))
        fx = run_fixed(process, params, x_te, cfg)
        parity = float(np.abs(torch_y - fl).max())
        # best LEGITIMATE naive predictor: training mean or the last
        # observed input. (Persistence on the target itself is excluded -
        # at forecast horizon h the previous target is h-1 steps in the
        # future, so that "baseline" would peek.)
        base = min(nrmse(np.full_like(y_te, y_tr.mean()), y_te),
                   nrmse(x_te[:, 0], y_te))
        m_fl, m_fx = nrmse(fl, y_te), nrmse(fx, y_te)
        metric, delta = "nrmse", m_fx - m_fl

    assert parity < 1e-4, f"IR/torch parity broke: {parity}"
    row = {"task": task_name, "model": model_name, "metric": metric,
           "float": round(m_fl, 4), "deployed_q6_12": round(m_fx, 4),
           "delta": round(delta, 4), "baseline": round(base, 4),
           "parity_torch_vs_ir": parity, "n_test": len(y_te),
           "train_s": round(time.time() - t0, 1),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    RESULTS.mkdir(exist_ok=True)
    with RESULTS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"[{task_name}/{model_name}] {metric}: float {m_fl:.4f} -> "
          f"deployed {m_fx:.4f} (baseline {base:.4f}, parity {parity:.1e})")
    return row


def print_report():
    if not RESULTS_FILE.exists():
        print("no results yet")
        return
    rows = [json.loads(line) for line in
            RESULTS_FILE.read_text(encoding="utf-8").splitlines()]
    latest = {}
    for r in rows:
        latest[(r["task"], r["model"])] = r
    print(f"{'task':<9}{'model':<9}{'metric':<10}{'float':>8}{'deployed':>10}"
          f"{'delta':>8}{'baseline':>10}")
    for (t, m), r in sorted(latest.items()):
        print(f"{t:<9}{m:<9}{r['metric']:<10}{r['float']:>8}"
              f"{r['deployed_q6_12']:>10}{r['delta']:>8}{r['baseline']:>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="mackey,narma,ecg")
    ap.add_argument("--models", default="gru,mingru,diag")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        print_report()
        return
    for t in args.tasks.split(","):
        for m in args.models.split(","):
            try:
                evaluate(t.strip(), m.strip())
            except Exception as exc:
                print(f"[{t}/{m}] FAILED: {exc}")
                with RESULTS_FILE.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"task": t, "model": m,
                                         "error": str(exc)[:300]}) + "\n")
    print()
    print_report()


if __name__ == "__main__":
    main()
