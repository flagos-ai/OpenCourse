# Day 2 Lab 5 - English Demonstration Script

**Duration:** approximately 30 minutes, including commands, output inspection and the optional recorded ixkn example. Omitting pages 14-15 shortens the walkthrough by about four minutes.

**Version:** aligned with README and Step 04 acceptance at `8f65814` (16 October 2026). Ixsmi and ixsys are required; ixkn is optional for acceptance.

**Companion PDF:** Day 2 Lab 5 demonstration, included in the [second-cohort archive release](https://github.com/flagos-ai/OpenCourse/releases/tag/edition-ai-compute-faculty-international-2nd-cohort-global). Page numbers below match the PDF.

**Historical reference:** the 15 October snapshot and six-process study are cited for context only; the source archive is not distributed with this course package. Pages 14-15 use a historical successful capture.

**Fault-server verification:** the 16 October fresh-clone reproduction and debugger evidence are cited for context only; the source archive is not distributed with this course package. Page 17 uses this run: ixkn returned -11 and Step 04 passed with a warning.

No manual implementation-list edit is required. Historical tables, the six-process study, and the fault-server rerun remain distinct evidence sets.

## Before class

Clone the current course repository before class, then work from `OpenCourse/labs/modules/module-05/iluvatar-corex-bi-v150`. Benchmark commands overwrite results in the active checkout; retain any evidence archives that you need.

```bash
git clone https://gitlink.org.cn/flagos-ai/OpenCourse.git
cd OpenCourse/labs/modules/module-05/iluvatar-corex-bi-v150
```

If matplotlib is missing, use the README installation command and refresh the environment check. Keep platform PyTorch and Triton versions.

Step 03 still automatically attempts ixkn and returns nonzero if its capture fails. If only ixkn fails while ixsmi and ixsys are captured, run Step 04 separately. Do not chain the two steps with `&&`. Failures of required evidence must still be resolved.

The optional autotune replay adds `--out results/autotune/diagnosis.json` so both evidence groups are retained. This is separate from ixkn being optional for acceptance. Ixkn is not skipped by default.

Pages 14-15 can be taught from the labeled historical capture or omitted. A live ixkn failure does not produce those counters. Do not force a device index to suppress a profiler crash; the device node number need not match the logical CUDA index.

Allow time for JIT and profiler startup. Keep the saved captures available during the approximately 30-minute explanation.

## 00:00-02:00 - Introduce the experiment

**PDF page:** 01

**Demonstration:** Show the workflow from the Module 5 lab directory.

**Spoken script:**

Today we will walk through a complete performance measurement workflow for matrix multiplication on the BI-V150 GPU.

We will compare PyTorch, a fixed Triton baseline, and Triton autotune. Our goal is to run the experiment and understand what the output means.

There are three questions. Which implementation completes a request faster? What changes when requests arrive over time? And what can profiling tell us about the performance difference?

We will check the environment, run Offline and Server benchmarks, inspect a Roofline model, collect profiling evidence, and validate the final results.

The main reference tables and counters come from the preserved October fifteenth snapshot. Six additional processes per scenario assess repeatability. Page seventeen shows a fresh clone on the affected server on October sixteenth, where ixkn crashed but the required captures and updated acceptance checks passed. We keep these evidence sets separate.

Throughout the demonstration, ask where the timer starts and stops. A GPU kernel duration, a request service time, and a request response time describe different intervals. Understanding that boundary is essential before comparing numbers.

## 02:00-05:00 - Check the environment

**PDF page:** 02

**Demonstration:** From /root, clone into an empty destination and run setup and smoke commands. Point to GPU visibility and the final smoke result.

```bash
cd OpenCourse/labs/modules/module-05/iluvatar-corex-bi-v150
source env_corex.sh
bash setup.sh
python3 00_check_env.py
python3 00_smoke_cuda.py
```

**Spoken script:**


The setup script checks the platform packages and selects the GPU. The recorded environment uses Python 3.12.11, PyTorch 2.10.0, and Triton 3.6.0.

Although the accelerator is an Iluvatar GPU, PyTorch exposes it through the CUDA interface. Seeing cuda:0 here is expected. The logical runtime index should not be inferred from a device filename.

The runtime version can be plain 2.10.0 while the wheel metadata contains the CoreX suffix. The environment checker considers that metadata, so the plain version alone does not indicate an incorrect package.

Now wait for the smoke test to finish with smoke OK. That confirms that a small allocation and matrix multiplication completed. It does not yet validate the larger Triton implementations or establish their performance.

Notice synchronization in the workflow. GPU submission can be asynchronous. If we stop a host timer before the submitted work completes, we may measure dispatch rather than completed computation. The benchmark synchronizes to include completion in its service time.

## 05:00-06:00 - Run Offline with completed defaults

**PDF page:** 03

**Demonstration:** Run the command. Point to both PASS messages and the three result rows.

```bash
python3 01_benchmark.py --scenario offline --M 1024 --N 1024 --K 1024 --repeats 3
```

**Spoken script:**

All three implementations are already enabled. We do not need to edit a TODO or pass an implementation override for this demonstration.

PyTorch is our vendor-library reference. The baseline uses a fixed Triton configuration. Autotune selects from predefined candidates; it does not search every possible implementation.

Run the Offline command for a 1024-cubed FP16 GEMM with three repeats. Each round contains sixteen requests, giving forty-eight measured samples per implementation.

Look for baseline PASS and autotune PASS. PyTorch does not print a Triton validation message. Then confirm that torch, baseline, and autotune each produce a metric row, followed by the final saved-file message.

SUT ready alone is not completion. The first Triton launch may still compile. Correctness validation, the search, and runner warmup happen outside the measured rounds.

## 06:00-08:00 - Read the original Offline table

**PDF page:** 04

**Demonstration:** Open comparison in benchmark_offline.json. Read the table and distinguish median from p99.

**Spoken script:**

These are the values saved in the original server results, rather than a selected best repeat.

PyTorch has a service median of 0.0749 milliseconds. The fixed baseline has 0.3326 milliseconds, and autotune has 0.2224 milliseconds.

Median describes the middle of the measured distribution. The p99 estimates a high percentile near the slow end. With only forty-eight samples, it is exploratory rather than a strong estimate of rare slow requests.

Samples per second divides completed queries by measured wall time. Offline drains the ready requests continuously, so this is useful throughput for this synchronized harness. It is not necessarily the maximum throughput under a different batching or concurrency strategy.

The autotuned implementation is about 1.50 times faster than the baseline here. Its service time remains about 2.97 times PyTorch's. The search improved this Triton implementation but did not close the vendor-library gap.

The JSON also records the selected configuration. The original Offline winner is 128 by 64 by 32, with four warps and two stages. Keep that configuration with the reported result.

## 08:00-10:00 - Explain the timing boundary

**PDF page:** 05

**Demonstration:** Point to the formulas and explain why kernel duration can be shorter than service time.

**Spoken script:**

Service latency starts when the worker begins dispatching the query and ends after synchronized completion. Host dispatch and synchronization are included. This is not a measurement of the GPU kernel alone.

For GEMM, the useful operation count is approximately two times M times N times K. Multiplication and addition count as two operations. At this shape, one query represents about 2.147 billion floating-point operations.

The script divides that work by median service time to obtain effective TFLOP/s. Convert milliseconds to seconds before doing the division.

Because this denominator includes host overhead, the effective rate can be below the rate calculated from a kernel-only duration. Both measurements can be correct within their timing scopes.

Also, samples per second is calculated from measured wall time, whereas effective TFLOP/s here uses median service time. One should not assume that multiplying the reported samples per second by the operation count will reproduce the median-based TFLOP/s exactly.

Warmup occurs before the measured rounds. Repeats reset the request schedule and pool the completed samples, while per-round statistics are preserved. They help us inspect variation rather than relying on a single short timing interval.

## 10:00-12:00 - Check Offline repeatability

**PDF page:** 06

**Demonstration:** Show all six points, the original dashed line, and the CV table. Allow a short comparison pause.

**Spoken script:**

We repeated the complete benchmark in six new processes. Each process retained three rounds and forty-eight samples per implementation.

We used all six permutations of implementation order, so each implementation appeared first, second, and third equally often. No profilers ran during these timings. Compilation caches were already warm.

Each panel has its own vertical scale. This makes small timing variations visible, so compare the numeric axes before judging the size of a movement.

The table summarizes the six process-level service medians. It shows their median, minimum-to-maximum range, and coefficient of variation. CV is sample standard deviation divided by the mean, expressed as a percentage. These CV values describe run medians, not the spread of individual query latencies.

The Offline CV values are torch: 1.11%, baseline: 5.03%, autotune: 0.43%. The baseline's higher run, at 0.3619 milliseconds, is retained in this summary rather than removed. Inspect the recorded configuration counts in the appendix as well: if autotune changes its winner, that variation includes selection behavior.

The original result is displayed separately as a dashed line. We keep every repeat, rather than selecting the fastest result. This supports a repeatability assessment for this machine and workload, not a universal performance guarantee.

## 12:00-13:00 - Run Server and distinguish the intervals

**PDF page:** 07

**Demonstration:** Run the command and point to the timeline.

```bash
python3 01_benchmark.py --scenario server --repeats 3
```

**Spoken script:**

The Server scenario changes how requests arrive. The default configuration uses Poisson arrivals with a target rate of eight queries per second.

This teaching harness has one synchronized worker. Requests can wait before that worker begins handling them.

Queue latency runs from scheduled arrival to service start. It includes waiting and host scheduling delay. Service latency runs from service start to synchronized completion. Response latency covers the whole interval.

For each request, response equals queue plus service. However, separate percentile statistics do not generally add. A queue median plus a service p99 will not necessarily equal response p99.

There are thirty-two requests per round and three rounds, giving ninety-six measured samples per implementation.

## 13:00-15:00 - Read the original Server table

**PDF page:** 08

**Demonstration:** Point to nearly equal samples/s, then inspect sla.timing_basis and sla.passed in the JSON.

**Spoken script:**

The original baseline service median is 0.4945 milliseconds. Its queue median is 0.1008 milliseconds, and response p99 is 0.6795 milliseconds.

Notice that the three throughput values are all about nine queries per second, despite the different service times. This does not establish equal processing capacity.

Consider the arrival rate. At this low offered load, the worker spends much of its time waiting for the next scheduled request. Throughput mainly reflects that schedule. A finite Poisson realization does not have to equal exactly the configured long-run target of eight.

The Server SLA uses response latency, because waiting is part of a request's experience. The current limit is five hundred milliseconds, and all three original results passed that threshold.

A passing SLA does not mean that the implementations perform equally well, or that the threshold is suitable for a real application. To investigate sustainable capacity, increase the offered rate and observe throughput, queue growth, and response latency together.

When comparing Offline and Server numbers, remember that the arrival pattern changes idle periods and host scheduling. The two scenarios can also use different autotune candidate sets, so always check the recorded configuration.

## 15:00-17:00 - Distinguish median repeatability from tail variation

**PDF page:** 09

**Demonstration:** Compare the service-median CV with response-p99 CV. Do not describe every metric as stable.

**Spoken script:**

This page shows response p99 from all six additional Server processes, with the original result as the dashed reference.

The service-median CV values are torch: 2.72%, baseline: 1.19%, autotune: 1.44%. The response-p99 ranges and CV values are shown separately because tail behavior is more sensitive to occasional delays.

Each process still contains only ninety-six samples. The six repeated processes provide five hundred and seventy-six samples per implementation in total, but we do not treat the six p99 estimates as a new guaranteed service-level bound.

The original PyTorch response p99 was 0.2854 milliseconds, while the repeats range from 0.3653 to 0.4147. That difference is visible on the chart and is another reason not to promise the exact same tail number in a live demonstration.

The arrival generator uses the same fixed seed and schedule across runs. This improves comparability but does not explore every arrival pattern or establish independent workload diversity.

We retain all runs and report their ranges. The appropriate conclusion distinguishes a repeatable ordering or median from a potentially noisier high percentile. A low-load throughput that barely changes can coexist with changing response p99.

For a stronger tail claim, collect longer runs and more request samples, vary arrival realizations deliberately, and inspect scheduler and system conditions. These short runs are suitable for explaining the measurement process and showing why medians and tails should be assessed separately.

## 17:00-19:00 - Read the Roofline

**PDF page:** 10

**Demonstration:** Run the script and open roofline.png. Show the axes, rising line, flat roof and ridge.

```bash
python3 02_roofline.py
```

**Spoken script:**

The Roofline plot relates arithmetic intensity to a reference performance ceiling. Both axes use logarithmic scales.

The horizontal axis is useful floating-point operations per modeled byte. The vertical axis is effective TFLOP/s from the Offline benchmark.

The rising line is bandwidth times arithmetic intensity. The flat line is the reference compute peak. The model ceiling is the lower of those two limits.

The current inputs are 192 TFLOP/s and 1228.8 gigabytes per second. They are explicitly marked as unverified assumptions for this instance. The ridge at 156.25 follows mathematically from those inputs; it does not verify the specifications.

The three points have the same modeled horizontal coordinate but different measured heights. Autotune improved the baseline, while the vendor library remained higher.

Also consider the measurement boundary. Our points use service time with host dispatch and synchronization, while the reference roof represents device capability. The distance from a point to the roof cannot be interpreted as a pure kernel-efficiency loss without accounting for that difference.

## 19:00-21:00 - Explain equal intensity and the compute-side label

**PDF page:** 11

**Demonstration:** Point to the equations and explain why compute-side does not establish compute saturation.

**Spoken script:**

For this FP16 GEMM, the simple model counts reading A and B and writing C once. That gives approximately six mebibytes of modeled traffic and an arithmetic intensity of 341.33.

The three implementations have the same matrix dimensions and element size, so their modeled arithmetic intensity is identical. Real traffic can differ because of reuse, cache behavior and repeated transfers. Those implementation-specific effects are not captured by this common horizontal coordinate.

Since 341.33 is greater than the assumed ridge of 156.25, all three points are on the compute side of this model. That is the correct interpretation of compute-side.

It does not establish that the measured kernel is limited by compute throughput. Register pressure, shared-memory behavior, scheduling, instruction overhead and other effects can keep it well below the reference roof.

Our next step is to collect evidence that can distinguish some of these explanations. We should treat the model as a guide for questions, while keeping actual bottleneck undetermined until the measurements and experiments support a narrower conclusion.

## 21:00-22:00 - Required and optional profiling captures

**PDF page:** 12

**Demonstration:** Run the baseline capture. Inspect ixsmi and ixsys first. If only ixkn is NOT CAPTURED, continue to Step 04 separately. Autotune replay is an additional optional capture.

```bash
python3 03_profiling.py --capture

# Optional replay of the selected autotune configuration:
python3 03_profiling.py --impl autotune --capture --out results/autotune/diagnosis.json
```

**Spoken script:**

Run the README capture command first. It selects baseline and replays the recorded shape, dtype, tile, warps and stages. SUT means System Under Test: here, the GEMM implementation and its configuration. The second command is an optional autotune replay. Its output path keeps the two evidence groups separate, and it does not search again under the profiler.

There are two required capture results: ixsmi and ixsys. Ixkn is optional for acceptance. This version of the wrapper still attempts all three tools automatically; optional does not mean ixkn is skipped by default.

On the affected server, ixsmi and ixsys show CAPTURED, followed by NOT CAPTURED for ixkn-cli. The recorded return code is minus eleven, indicating a segmentation fault. Step three returns a nonzero exit code, but it still writes the diagnosis and capture metadata for this failure.

If ixkn is the only failed tool, run Step four separately. Do not stop solely because of that ixkn message. If ixsmi or ixsys failed, their required evidence still needs to be repaired.

The final model message has a different meaning. Compute-side describes the Roofline region. Actual bottleneck undetermined is expected and is not a capture error. A tool being available only means its executable was found. CAPTURED means the expected evidence was produced and checked.

## 22:00-23:00 - Read activity and trace evidence

**PDF page:** 13

**Demonstration:** Use the historical activity figure, then inspect the live ixsmi and ixsys capture metadata. These captures are required.

**Spoken script:**

The utilization sampler collected thirteen samples. SM activity reached one hundred percent during the sustained workload, with idle samples around it.

This shows that the GPU was busy during the sample interval. It does not mean that the workload achieved one hundred percent of peak arithmetic performance. A busy device can still execute an inefficient kernel.

The ixsys trace contains twenty-four records for our GEMM kernel. That confirms that expected GPU work was captured. It is not a throughput measurement. The trace includes launches associated with warmup and validation as well as the requested workload iterations.

This chart uses the October fifteenth capture. The fault-server run also captured ixsmi and ixsys successfully. Required activity and trace evidence remain available when ixkn fails. The next two pages are an optional example from a separately preserved successful ixkn capture, not measurements from the failed run.

## 23:00-25:00 - Optional: kernel counter example

**PDF page:** 14

**Demonstration:** Use the archived successful autotune capture. If live ixkn fails, use this labeled reference or skip pages 14-15; an ixkn capture is not required for completion.

**Spoken script:**

This page is optional. Its values come from the successful October fifteenth autotune capture. They are not measurements recovered from the failed fault-server capture.

The original kernel duration is 0.1277 milliseconds. The original Offline service median is 0.2224 milliseconds. These cover different intervals. The service measurement includes host dispatch and synchronization, and profiler replay can alter execution conditions. Their difference is not an exact overhead decomposition.

Achieved occupancy is 24.55 percent, close to the reported theoretical occupancy of 25.00 percent. Occupancy describes active warps relative to the reported hardware limit, not a fraction of peak floating-point speed.

TCU efficiency is a vendor-reported tensor-compute indicator. LSU utilization concerns the load/store unit. A high LSU value gives us a reason to investigate data movement, but it does not prove that external HBM bandwidth is saturated.

The L2 cache load hit rate is 95.01 percent. Many counted loads hit in L2, but that alone does not establish optimal memory behavior. Access volume, latency and on-chip transfers still matter.

Finally, shared-memory stores report 1,859,584 bank conflicts. This is a useful lead for a follow-up experiment, rather than a complete explanation of the vendor-library gap.

## 25:00-27:00 - Optional: interpreting kernel counters

**PDF page:** 15

**Demonstration:** Explain the archived counter example and the device-mapping failure. No kernel edit or debugger operation is required from students.

**Spoken script:**

This optional interpretation uses the archived successful capture. It shows ninety-two vector registers per thread and identifies VRF requirements as a theoretical occupancy limiter. Register pressure deserves investigation, but increasing occupancy does not guarantee a proportional speedup. A different tile changes reuse and instruction overhead as well as register use.

Shared-memory bank conflicts suggest an access-layout hypothesis. A warning count alone does not show how much of the total time it explains. A follow-up would change one factor, validate numerical output, and compare timings without the profiler.

The affected container exposes device node iluvatar thirteen, while the only visible GPU is logical cuda zero. These are different numbering systems. The runtime uses the correct logical device. The debugger located the crash in ixkn's device-type lookup during report postprocessing, after GEMM replay completed.

The tool mishandles that physical-to-logical mapping. A debugger-only mapping correction produced a correctly identified report, but it was a diagnostic experiment, not a deployed repair. Forcing CUDA_VISIBLE_DEVICES to zero avoided the crash while producing incorrect device-family metadata, so it is not a reliable fix.

For the class, keep ixkn optional for acceptance and preserve its warning. Reliable kernel-level collection on the affected layout needs a tool-level fix. The actual bottleneck remains undetermined.

## 27:00-28:00 - Validate the result set

**PDF page:** 16

**Demonstration:** Run Step 04 separately after the capture attempt. Open summary.json and distinguish the optional ixkn warning from required-evidence failures.

```bash
python3 04_summary.py
```

**Spoken script:**

The summary requires all three benchmark implementations in both scenarios, numerical validation, compatible workload information, valid provenance, passing configured SLA checks, and captured ixsmi and ixsys artifacts.

Ixkn is optional for acceptance. Missing or failed ixkn evidence, including return code minus eleven, produces a warning rather than failing the summary by itself. Successful ixkn artifacts are still checked. An invalid optional report also produces a warning instead of being treated as verified counter evidence.

The real fault-server run demonstrates this path. Step three reported NOT CAPTURED for ixkn and returned one. Step four then printed the optional-ixkn warning and acceptance PASS, with an empty errors list. This is possible because the required results were valid. A required capture failure, failed numerical validation, or stale result still fails acceptance.

Read the heading carefully: service median and response p99. Offline response includes waiting behind earlier requests and is not one GEMM's duration. Small-sample and unverified-peak warnings also remain.

PASS confirms the required automated checks. It does not claim that optional ixkn counters were collected, certify peak specifications or stable tails, or identify the actual bottleneck.

## 28:00-30:00 - Verify completion on the fault server

**PDF page:** 17

**Demonstration:** Open the fault-server archive: results/summary.json and results/reproduction-verification.json. Show the warning, empty errors list, PASS status, and benchmark table.

**Spoken script:**

The walkthrough ends by checking the generated results. Both benchmark files must contain torch, baseline and autotune. The Roofline JSON and image, diagnosis, and required ixsmi and ixsys artifacts must be present and consistent. The summary status is PASS and its errors list is empty.

The affected server was tested from a fresh clone of commit 8f65814. Environment checks, smoke, Offline, Server and Roofline completed. Ixsmi and ixsys captured successfully. Ixkn reproduced the segmentation fault with return code minus eleven. The updated summary passed with an explicit warning. All twenty-four regression tests passed.

This fault-server run has Offline service medians of 0.0758 milliseconds for PyTorch, 0.3348 for baseline, and 0.2316 for autotune. The table also shows Server service medians. These are one run's results with three rounds per scenario, separate from the historical six-process study.

The performance ordering agrees with the historical example: autotune improves the fixed baseline, while PyTorch remains faster. All thirty-six tracked source files stayed unchanged. Forty-eight existing files in the original checkout were also verified unchanged.

Completion uses the required measurement and profiling evidence. Optional kernel counters can support further analysis when reliably available; their absence remains visible. The actual bottleneck is undetermined. Small-sample and unverified-peak interpretation limits remain.

## Appendix - PDF pages 18-20

- **Page 18:** all six historical process-level service medians.
- **Page 19:** historical Server response-p99 ranges and selected autotune configurations.
- **Page 20:** historical sources and the fault-server verification archive.

Use the appendix as reference while explaining the outputs.
