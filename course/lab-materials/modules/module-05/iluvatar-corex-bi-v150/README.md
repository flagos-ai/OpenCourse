# Module 5: Performance Measurement, Roofline, and Profiling

Repository: [lab5-profiler](https://gitee.com/sunxt-0719/lab5-profiler)

**Course:** AI Systems Software & Heterogeneous Computing  
**Lab length:** ~2.5–3 hours on BI-V150  
**Platform:** Iluvatar BI-V150 (FlagOS / CoreX)  
**Outputs:** `results/*.json`, optional `roofline.png`, and profiling artifacts

---

## Framing

Day 1–2 built a correct tiled Triton GEMM and practiced trustworthy timing
(warmup, sync, median). A recurring observation on BI-V150 is that a tuned
Triton GEMM may still lag `torch.matmul` (vendor library).

This module does **not** ask you to invent a new kernel. It asks:

> How do we measure fairly, interpret a Roofline model,
> and use vendor profilers to investigate the measured bottleneck?

Method (same as Day 2):

> **Hypothesis → Experiment → Data → Refined understanding**

---

## Learning Goals

After this lab you should be able to:

1. Run Offline and Server-style LoadGen workloads and report latency / throughput / TFLOP/s.
2. Compute GEMM arithmetic intensity and place implementations on a Roofline.
3. Use `ixsmi`, `ixsys`, and `ixkn-cli` at a basic level (Proton is not supported on Iluvatar).
4. Separate **document claims**, **code defaults**, and **measured facts**.

---

## Layout

```text
lab5-profiler/
  setup.sh  env_corex.sh
  00_check_env.py  00_smoke_cuda.py
  01_benchmark.py  02_roofline.py  03_profiling.py  04_summary.py
  configs/student_todos.py   # implementation defaults ready; Roofline peak exercise
  kernels/  loadgen/  tools/ # completed helpers — do not edit
```

Lab1's GEMM is integrated directly in `kernels/gemm_triton.py`; no sibling checkout or external path is required. The native `triton.autotune` path follows Lab1 Step 08 and takes candidates from this repository's JSON configs. Candidate and selected-output correctness checks run outside LoadGen timing. See [`docs/GEMM_SOURCE.md`](docs/GEMM_SOURCE.md) for the exact upstream revision and mapping.

---

## Step 00: Environment and CUDA smoke (15 min)

```bash
cd labs/module-05
source env_corex.sh
bash setup.sh
python3 00_check_env.py
python3 00_smoke_cuda.py    # must finish with [smoke] OK on FlagOS
```

### Notes

- `/dev/iluvatarN` is a **device node name**, not necessarily the CUDA index.
  Inside a one-GPU container the runtime usually still exposes **logical `cuda:0`**.
  Do **not** set `CUDA_VISIBLE_DEVICES=N` just because you see `iluvatarN`.
- If native `bandwidthTest` hangs as well as PyTorch, the instance is broken —
  restart or contact the TA (not a Module-5 code bug).

---

## Step 01: Benchmark / LoadGen (40 min)

Both scenarios already enable `torch`, `baseline`, and `autotune` in
[`configs/student_todos.py`](configs/student_todos.py), mirrored in the JSON
configs. No implementation-list edit or `--impl` override is needed for the
three-way comparison. The first Triton JIT on CoreX can be slow.

```bash
python3 01_benchmark.py --scenario offline --M 1024 --N 1024 --K 1024 --repeats 3
python3 01_benchmark.py --scenario server --repeats 3
```

If FlagOS lacks matplotlib:

```bash
python3 -m pip install matplotlib
```

`02_roofline.py` still writes JSON without PNG when matplotlib is missing.

Outputs: `results/benchmark_offline.json`, `results/benchmark_server.json`.

### Metrics

| Metric | Meaning |
|--------|---------|
| service latency (ms) | Dispatch to synchronized completion, including host overhead; `latency_ms` remains an alias |
| queue latency (ms) | Scheduled arrival to service start, including scheduling delay |
| response latency (ms) | Queue + service; **Server SLA uses response latency** |
| samples/s | Completed queries divided by the sum of measured round wall times |
| effective TFLOP/s | `2*M*N*K / (median_service_time_s) / 1e12` |

This is a teaching LoadGen with one synchronized worker, not an official MLPerf
submission. Server requests follow the configured arrival schedule; requests
arriving while the worker is busy wait in its queue. Low-QPS throughput reflects
the arrival schedule, not maximum GPU capacity. Offline issues all requests at
round start; use service latency for its per-query performance comparison.

`--repeats` overrides `measurement.repeats` (default 1). Each round restarts the
arrival schedule; warmup runs once before the rounds. Statistics pool
`num_queries * repeats` completed samples; `repetitions` retains per-round
statistics. `sync=false` is rejected because asynchronous dispatch time would
not measure completion. The default 16/32-query runs, even with three rounds,
are demonstrations; fewer than 100 samples produce an exploratory-p99 warning.
Use more queries and inspect variability before making tail-latency claims.

---

## Step 02: Roofline (30 min)

In `configs/student_todos.py`, verify:

```python
PEAK_TFLOPS_FP16 = ...    # π
HBM_BANDWIDTH_GBPS = ...  # β
```

The shipped values (192 TFLOP/s and 1228.8 GB/s) are **unverified reference
assumptions**, not confirmed specifications for this instance. Record a source
before treating the roof as a hardware limit. By default, positive TODO values
override `configs/hardware_bi_v150.json`; keep the mirror consistent. An explicit
`--hardware PATH` instead uses that JSON's values as authoritative. The output
records parameter provenance and mirror agreement. `peaks_verified` is true only
when the JSON declares it, supplies `source_reference`, and its values match the
selected values. Update these fields only after checking the specifications.

```bash
python3 02_roofline.py
```

Outputs: `results/roofline.json`, `results/roofline.png`.

For GEMM:

```text
FLOPs  = 2 * M * N * K
Bytes  ≈ (M*K + K*N + M*N) * sizeof(dtype)   # simple traffic model
I      = FLOPs / Bytes
P_roof = min(π, β * I)
```

The JSON records `model_region` as `compute-side` or `memory-side` and keeps
`actual_bottleneck` as `undetermined`. The simplified traffic model and
host-inclusive service timing cannot establish a hardware bottleneck.

---

## Step 03: Profiling (40 min)

Run the capture wrapper after Step 02:

```bash
python3 03_profiling.py --capture
```

It reads `results/benchmark_offline.json`, selects the validated **baseline**,
and replays its recorded shape, dtype, tile, warps, and stages. No manual shape
alignment is needed. To inspect the autotune winner instead:

```bash
python3 03_profiling.py --impl autotune --capture
```

The winner is replayed as a fixed configuration; profiling does not rerun the
candidate search. Diagnosis matches this implementation's Roofline entry.

`--capture` runs a sustained workload (default 5 seconds) alongside `ixsmi dmon`,
collects an ixsys trace, and exports/imports one ixkn GEMM report. The kernel
filter skips three warmup launches. Outputs are under `results/profiling/`:
`ixsmi_dmon.log`, `gemm.trace`, `gemm_kernel.ixkn-rep`, tool logs, and
`profiling_summary.json`; `results/diagnosis.json` joins their metadata to the
matching Roofline entry. Capture verifies nonempty artifacts, visible GEMM
activity, and tool exit status. Missing tools or failed captures return nonzero.

For Step 04 acceptance, **ixsmi and ixsys are required; ixkn is optional**.
If only ixkn fails (including a profiler crash), Step 03 currently still returns
nonzero, but it writes the diagnosis and capture metadata. Run Step 04 separately
to validate the remaining evidence. The summary can pass with an explicit ixkn
warning; this does not claim that kernel-level counters were collected.

Running without `--capture` only executes the workload and discovers tools:
`available` does **not** mean `captured`, and Step 04 rejects this incomplete
result. `--workload-only --duration 5` is available for manual experiments;
manual captures do not create the validated manifest required by Step 04.
Proton is unsupported and `--enable-proton` reports an error.

Profiler replay changes timing. Use the unprofiled LoadGen results for performance
comparisons. Utilization/occupancy alone also do not prove a specific bottleneck.

| Tool | Role |
|------|------|
| `ixsmi` / `ixsmi dmon` | Utilization, memory, power |
| `ixsys` | System / CUDA API-level trace |
| `ixkn-cli` | Optional kernel-level instruction / occupancy style views |
| Proton | **Not supported** on Iluvatar — skip or note “N/A” |

---

## Step 04: Summary (15 min)

```bash
python3 04_summary.py
```

The command writes `results/summary.json` and returns **nonzero** for missing,
malformed, stale, or inconsistent results, failed numerical validation, failed
configured SLA, or missing/changed required profiler artifacts. By default, both scenarios
must include `torch`, `baseline`, and `autotune`. For an intermediate two-way run,
use `--required-impl torch baseline`.

Required profiler evidence is **ixsmi and ixsys**. Missing, failed, or invalid
ixkn evidence produces a warning instead of failing acceptance. A successful ixkn
capture is still checked against its artifact hash; if that check fails, the
summary warns that its kernel-level counters are not validated. All benchmark,
numerical validation, SLA, provenance, and required-artifact checks remain active.
With otherwise valid results, an existing ixkn-only failure can be rechecked by
running Step 04 after updating the repository; no benchmark rerun is needed for
this acceptance-only change.

Results use schema version 2 and record benchmark IDs, code fingerprints, and
artifact hashes. After changing measurement code or regenerating a benchmark,
rerun Steps 01–03 before summarizing. Old JSON files must be regenerated. Keep
artifacts at their recorded paths when validating on the server.

`PASS` confirms these automated consistency/completeness checks. It does not
certify the peak specifications, stable p99, or the actual hardware bottleneck;
small-sample and unverified-peak warnings remain visible.

For maintainers, regression checks on the lab environment are:

```bash
python3 -m unittest discover -s tests -v
```

---

## What We Did Not Cover

- KernelGen / LLM-assisted codegen (optional lecture topic)
- Multi-GPU, quantized dtypes, persistent GEMM, fused epilogues

Change one variable at a time if you continue after class.

---

## Completion Checklist

- [ ] `setup.sh` / `00_check_env` / `00_smoke_cuda` OK on FlagOS
- [ ] Offline + Server benchmarks written under `results/`
- [ ] Both benchmark results contain `torch`, `baseline`, and `autotune`
- [ ] Hardware π / β source recorded, or explicitly labeled unverified assumptions
- [ ] Roofline JSON (and PNG when available) and profiling diagnosis generated
- [ ] `03_profiling.py --capture` captured required ixsmi and ixsys evidence; optional ixkn failures reviewed
- [ ] `04_summary.py` reports PASS; warnings have been reviewed

Reference solutions (if any) live in a teacher-managed `solution/` tree — not in this student repo.
