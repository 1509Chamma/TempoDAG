# Environment for the Vitis proof sprint

## What the human must install (Claude cannot do this part)

1. **AMD Vitis HLS** (2023.2 or newer recommended; Kria/Zynq targets need no
   paid license). Download via the AMD Unified Installer (~40-100 GB, free
   AMD account required): https://www.xilinx.com/support/download.html
   - Recommended: install under **WSL2/Ubuntu** — Vitis is happier on Linux
     and the harness's headless flow matches CI. Windows-native also works.
2. Ensure `vitis_hls` is on PATH for whatever shell runs the harness
   (e.g. `source /tools/Xilinx/Vitis_HLS/2023.2/settings64.sh` in WSL).

## Then the proof is one command per rung

```
python experiments/hls_proof/run_experiment.py --stage emit    # works today
python experiments/hls_proof/run_experiment.py --stage csim    # parity vs golden trace
python experiments/hls_proof/run_experiment.py --stage synth   # II/latency/resources
python experiments/hls_proof/run_experiment.py --stage cosim   # RTL-level confirmation
```

## Pinned choices (change knowingly)

- Part: `xck26-sfvc784-2LV-c` (Kria KV260). Edit `PART` in run_experiment.py
  for another board.
- Clock: 5 ns (200 MHz) — matches the demo's report assumption.
- `TOP_FUNCTION`: verify against the generated header on first csim run
  (known W1 task; the generator's top-name convention must be confirmed).

## Discovered on first contact (2026.1, Windows) — already handled by the harness

1. **No classic `vitis_hls` binary.** 2025.x/2026.x removed it; the harness
   uses the unified flow: `v++ -c --mode hls` (synth) and
   `vitis-run --mode hls --csim/--cosim` (simulation).
2. **Paths must be space-free** (HLS 200-2015). This repo lives under
   "Personal Projects", so all Vitis work runs in `C:\tmp\tempodag_hls`
   (override with TEMPO_HLS_WORKSPACE).
3. **A no-cost license must be ACTIVATED** (new requirement since 2025.x —
   even the free Standard tier). Symptom: "valid license was not found" +
   "Part ... not supported". Fix (human step, needs the AMD account):
   generate the free node-locked Vivado/Vitis Standard license at
   https://www.xilinx.com/getlicense, download `Xilinx.lic`, save it to
   `C:\Users\<you>\.Xilinx\Xilinx.lic`, re-run. The K26 part data is
   confirmed installed; only activation is missing.

## First-run expectations (honest)

The generated HLS has NEVER been through Vitis. Expect W1-class issues:
template gaps, type mismatches, II violations at the recurrent edge. That is
the point of the sprint — each failure is logged in results/*.log and fixed
in the templates, not papered over.
