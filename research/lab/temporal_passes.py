"""Prototype temporal graph passes + param-level models (NB04/NB05 validated).

Graph level:
  collapse_delay_chains  — Delay(a)->Delay(b) => Delay(a+b), bit-exact
                           (currently REJECTED by validate_temporal_rewrite;
                           needs the parity-certificate contract extension)
  share_nested_windows   — smaller rolling windows read the largest buffer
                           (annotation-only; legal under the current contract)

Param level:
  kernel_critical_path   — critical-path kernel latency, unroll-aware
  recurrence_aware_ii    — II floor incl. MII_rec = ceil(L_cycle / lag)
  bind_buffer_storage    — depth-based storage binding (SRL/LUTRAM vs BRAM)
"""

from __future__ import annotations

import math
import sys
from copy import deepcopy
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tempo_dag.ir_temporal import BufferSpec, Process  # noqa: E402
from tempo_dag.ir_temporal.schedule import derive_temporal_schedule  # noqa: E402
from tempo_dag.ops.temporal_builtins import Delay  # noqa: E402


# ------------------------- graph-level passes -------------------------
def collapse_delay_chains(process: Process) -> Process:
    """Apply Delay(a)->Delay(b) => Delay(a+b) to fixpoint (all kernels)."""
    current = deepcopy(process)
    changed = True
    while changed:
        current, changed = _collapse_once(current)
    return current


def _collapse_once(p: Process) -> tuple[Process, bool]:
    for kernel in p.kernels.values():
        g = kernel.graph
        consumers: dict[str, list[str]] = {}
        for oid, op in g.ops.items():
            for i in op.inputs:
                consumers.setdefault(i, []).append(oid)
        for oid, op in sorted(g.ops.items()):
            if op.op_type != "Delay":
                continue
            out = op.outputs[0]
            cons = consumers.get(out, [])
            if len(cons) != 1 or g.ops[cons[0]].op_type != "Delay" \
                    or out in g.graph_outputs:
                continue
            nxt = g.ops[cons[0]]
            lag = op.attrs.get("lag_cycles", 1) + nxt.attrs.get("lag_cycles", 1)
            keep = op.attrs.get("buffer_id", f"{oid}_buffer")
            fused = Delay(f"{oid}_{nxt.op_id}_fused", inputs=list(op.inputs),
                          outputs=list(nxt.outputs),
                          attrs={"lag_cycles": lag, "buffer_id": keep,
                                 "fused_ops": [oid, nxt.op_id]})
            del g.ops[oid], g.ops[nxt.op_id]
            g.ops[fused.op_id] = fused
            g.values.pop(out, None)
            drop = nxt.attrs.get("buffer_id")
            if keep in p.buffers and drop in p.buffers:
                old = p.buffers[keep]
                p.buffers[keep] = BufferSpec(
                    keep, old.dtype, old.shape, depth=lag, axes=old.axes,
                    clock_id=old.clock_id,
                    metadata={**old.metadata, "merged_from": [keep, drop]})
                del p.buffers[drop]
                p.edge0 = [e for e in p.edge0 if e.source != drop]
                p.edge_delta = [e for e in p.edge_delta if e.target != drop]
            return p, True
    return p, False


def share_nested_windows(process: Process) -> Process:
    """Annotate smaller same-input rolling windows to read the largest buffer."""
    p = deepcopy(process)
    for kernel in p.kernels.values():
        stats = [(oid, op) for oid, op in kernel.graph.ops.items()
                 if op.op_type in ("RollingMean", "RollingVar")]
        by_input: dict[str, list] = {}
        for oid, op in stats:
            by_input.setdefault(op.inputs[0], []).append((oid, op))
        for group in by_input.values():
            if len(group) < 2:
                continue
            largest = max(group, key=lambda kv: kv[1].attrs.get("window_size", 1))
            big = largest[1].attrs.get("buffer_id")
            for _oid, op in group:
                bid = op.attrs.get("buffer_id")
                if bid != big and bid in p.buffers:
                    old = p.buffers[bid]
                    p.buffers[bid] = BufferSpec(
                        bid, old.dtype, old.shape, old.depth, axes=old.axes,
                        clock_id=old.clock_id,
                        metadata={**old.metadata, "physical_buffer_id": big,
                                  "window_of": big})
    return p


# ------------------------- param-level models -------------------------
def kernel_critical_path(kernel, unroll: int = 1) -> int:
    """Critical-path latency over the kernel op DAG; unroll-aware elementwise."""
    g = kernel.graph
    producers = {o: op_id for op_id, op in g.ops.items() for o in op.outputs}

    def op_latency(op):
        if unroll == 1:
            return op.estimate_fpga_cost(g.values).latency_cycles
        out = g.values[op.outputs[0]]
        elems = math.prod(out.shape) if out.shape else 1
        return math.ceil(elems / unroll) + 1

    lat = {op_id: op_latency(op) for op_id, op in g.ops.items()}
    memo: dict[str, int] = {}

    def path(op_id):
        if op_id in memo:
            return memo[op_id]
        best = 0
        for i in g.ops[op_id].inputs:
            prod = producers.get(i)
            if prod is not None and prod != op_id:
                best = max(best, path(prod))
        memo[op_id] = best + lat[op_id]
        return memo[op_id]

    return max((path(o) for o in g.ops), default=1)


def recurrence_aware_ii(process: Process, unroll: int = 1,
                        samples_per_firing: int = 1) -> tuple[int, int, float]:
    """(naive_ii, recurrence_ii, per_sample_ii) — the NB04 model."""
    naive = derive_temporal_schedule(process).estimated_initiation_interval
    ii = naive
    for e in process.edge_delta:
        kids = {k for k in (e.source, e.target) if k in process.kernels}
        for e0 in process.edge0:
            if e0.source in (e.source, e.target) and e0.target in process.kernels:
                kids.add(e0.target)
            if e0.target in (e.source, e.target) and e0.source in process.kernels:
                kids.add(e0.source)
        L = sum(kernel_critical_path(process.kernels[k], unroll) for k in kids)
        ii = max(ii, math.ceil(L / e.lag_cycles))
    return naive, ii, ii / samples_per_firing


def bind_buffer_storage(process: Process, *, srl_max_depth: int = 32,
                        width_bits: int = 32) -> dict[str, dict]:
    """Depth-based storage binding for each PHYSICAL buffer.

    depth <= srl_max_depth -> shift register (FF/SRL, zero BRAM);
    otherwise -> BRAM (18kbit blocks). Buffers annotated with a
    physical_buffer_id are aliases and consume no storage of their own.
    """
    binding: dict[str, dict] = {}
    for bid, buf in sorted(process.buffers.items()):
        phys = buf.metadata.get("physical_buffer_id", bid)
        if phys != bid:
            binding[bid] = {"binding": f"alias->{phys}", "bram_blocks": 0,
                            "ff_bits": 0}
            continue
        elems = buf.depth * max(1, math.prod(buf.shape))
        bits = elems * width_bits
        if buf.depth <= srl_max_depth:
            binding[bid] = {"binding": "shift_register", "bram_blocks": 0,
                            "ff_bits": bits}
        else:
            binding[bid] = {"binding": "bram", "bram_blocks": math.ceil(bits / 18432),
                            "ff_bits": 0}
    return binding


def cse_rolling_stats(process: Process) -> Process:
    """Deduplicate identical (op_type, input, window) rolling stats to fixpoint.

    Bit-exact (identical computation removed); currently REJECTED by
    validate_temporal_rewrite because op/buffer sets change.
    """
    current = deepcopy(process)
    changed = True
    while changed:
        current, changed = _cse_once(current)
    return current


def _cse_once(p: Process) -> tuple[Process, bool]:
    for kernel in p.kernels.values():
        g = kernel.graph
        seen: dict[tuple, tuple] = {}
        for oid, op in sorted(g.ops.items()):
            if op.op_type not in ("RollingMean", "RollingVar"):
                continue
            key = (op.op_type, op.inputs[0], op.attrs.get("window_size", 1))
            if key not in seen:
                seen[key] = (oid, op)
                continue
            _keep_id, keep_op = seen[key]
            dup_out, keep_out = op.outputs[0], keep_op.outputs[0]
            for cid, cop in g.ops.items():
                if cid != oid:
                    cop.inputs[:] = [keep_out if i == dup_out else i
                                     for i in cop.inputs]
            g.graph_outputs[:] = [keep_out if o == dup_out else o
                                  for o in g.graph_outputs]
            drop_buf = op.attrs.get("buffer_id")
            del g.ops[oid]
            g.values.pop(dup_out, None)
            if drop_buf in p.buffers:
                del p.buffers[drop_buf]
                p.edge0 = [e for e in p.edge0 if e.source != drop_buf]
                p.edge_delta = [e for e in p.edge_delta if e.target != drop_buf]
            return p, True
    return p, False


def share_mean_into_var(process: Process) -> tuple[Process, int]:
    """Annotate RollingVar ops to consume a coexisting RollingMean's state.

    Uses var = E[x^2] - mu^2 with the SHARED mu. Annotation-only (contract-
    legal today). CAUTION: the shared form is cancellation-prone in fixed
    point (Welford vs naive variance); requires NB03-style format analysis
    before the cost credit is claimed in hardware.
    """
    p = deepcopy(process)
    annotated = 0
    for kernel in p.kernels.values():
        g = kernel.graph
        means = {(op.inputs[0], op.attrs.get("window_size", 1)): oid
                 for oid, op in g.ops.items() if op.op_type == "RollingMean"}
        for op in g.ops.values():
            if op.op_type != "RollingVar":
                continue
            key = (op.inputs[0], op.attrs.get("window_size", 1))
            if key in means and "shares_mean_with" not in op.attrs:
                op.attrs["shares_mean_with"] = means[key]
                annotated += 1
    return p, annotated


# ---------------- refined II attribution (NB09) ----------------
def _cycle_ops_for_edge(process: Process, e) -> dict[str, set[str]]:
    """Ops actually on the feedback cycle of one EdgeDelta, per kernel.

    Union of (a) path-based: ops on any path from the cycle's read value_id
    (Edge0 out of the state/buffer) to its write value_id (the EdgeDelta's),
    and (b) ownership-based: TemporalOperators whose temporal_metadata
    references the endpoint buffer/state id.
    """
    result: dict[str, set[str]] = {}
    endpoint_ids = {e.source, e.target}
    read_vals = {ed.value_id for ed in process.edge0
                 if ed.source in endpoint_ids and ed.value_id}
    write_vals = {e.value_id} if e.value_id else set()
    for kid, kernel in process.kernels.items():
        g = kernel.graph
        onpath: set[str] = set()
        if read_vals and write_vals:
            fwd: set[str] = set()
            frontier = set(read_vals)
            while frontier:
                nxt: set[str] = set()
                for oid, op in g.ops.items():
                    if oid not in fwd and any(
                            i in frontier or i in read_vals for i in op.inputs):
                        fwd.add(oid)
                        nxt.update(op.outputs)
                frontier = nxt
            back: set[str] = set()
            frontier = set(write_vals)
            while frontier:
                nxt = set()
                for oid, op in g.ops.items():
                    if oid not in back and any(
                            o in frontier or o in write_vals for o in op.outputs):
                        back.add(oid)
                        nxt.update(op.inputs)
                frontier = nxt
            onpath = fwd & back
        owns: set[str] = set()
        for oid, op in g.ops.items():
            meta_fn = getattr(op, "temporal_metadata", None)
            if meta_fn is None:
                continue
            try:
                meta = meta_fn(g.values)
            except Exception:
                continue
            refs = set(meta.buffers) | set(meta.state_reads) | set(meta.state_writes)
            if refs & endpoint_ids:
                owns.add(oid)
        cyc = onpath | owns
        if cyc:
            result[kid] = cyc
    return result


def _restricted_critical_path(kernel, op_subset: set[str]) -> int:
    g = kernel.graph
    producers = {o: oid for oid, op in g.ops.items() for o in op.outputs}
    lat = {oid: op.estimate_fpga_cost(g.values).latency_cycles
           for oid, op in g.ops.items() if oid in op_subset}
    memo: dict[str, int] = {}

    def path(oid):
        if oid in memo:
            return memo[oid]
        best = 0
        for i in g.ops[oid].inputs:
            prod = producers.get(i)
            if prod in op_subset and prod != oid:
                best = max(best, path(prod))
        memo[oid] = best + lat[oid]
        return memo[oid]

    return max((path(o) for o in op_subset), default=0)


def recurrence_aware_ii_v2(process: Process, unroll: int = 1,
                           samples_per_firing: int = 1) -> tuple[int, int, float]:
    """(naive_ii, recurrence_ii, per_sample_ii) with per-cycle op attribution.

    Unlike v1 (whole-kernel critical path per EdgeDelta), v2 charges only the
    ops actually on each feedback cycle; an EdgeDelta with no carried compute
    (e.g. a window buffer written with the raw input) contributes nothing.
    """
    del unroll  # cycle ops here are temporal/state updates; unroll handled by v1 callers
    naive = derive_temporal_schedule(process).estimated_initiation_interval
    ii = naive
    for e in process.edge_delta:
        cyc = _cycle_ops_for_edge(process, e)
        if not cyc:
            continue
        L = sum(_restricted_critical_path(process.kernels[k], ops)
                for k, ops in cyc.items())
        if L > 0:
            ii = max(ii, math.ceil(L / e.lag_cycles))
    return naive, ii, ii / samples_per_firing


__all__ = ["bind_buffer_storage", "collapse_delay_chains", "cse_rolling_stats",
           "kernel_critical_path", "recurrence_aware_ii",
           "recurrence_aware_ii_v2", "share_mean_into_var",
           "share_nested_windows"]
