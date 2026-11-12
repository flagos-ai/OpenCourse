# FlagOS-FlagCompressor Quantization Lab Manual

> For beginners: Complete a hands-on LLM quantization experiment by manually entering commands in the online lab terminal.
> You will perform model quantization, validate the output, attempt inference loading, and observe the differences before and after quantization.

## Lab Objectives

- Understand quantization principles (INT8/INT4, group/channel strategies, MSE search)
- Master the full FlagOS-Compressor workflow: inspect → dry-run → quantize → validate
- Observe changes in model size and GPU memory before and after quantization
- Attempt to load quantized outputs with vLLM and observe current inference framework support

## Chapter 0: Environment Assessment (Read This First)

### 0.1 Your Lab Environment

| Item | Value |
|---|---|
| Chip | Iluvatar BI-V150 ×4 cores |
| Memory per core | 32 GiB |
| Driver / IX-ML | 4.5.0 / 4.4.0 |
| CUDA compatibility layer | 10.2 |
| Compute capability | **7.1** (critical — see §0.3) |
| Python | 3.12 |
| PyTorch | 2.10.0+corex |
| vLLM | 0.24.0+empty |
| Model to quantize | `/models/Qwen3.6-35B-A3B/` (pre-loaded, BF16, 67G, multimodal MoE) |

### 0.2 Verify Your Environment: First Command

In the terminal, type (this is the Iluvatar GPU monitor, equivalent to NVIDIA's nvidia-smi):

```bash
ixsmi
```

You should see 4 BI-V150 GPUs, each with 32768 MiB of memory. Then type:

```bash
hostname -I
```

Note the IP address shown (e.g. `172.17.0.2`) — you will need it when writing test scripts later.

### 0.3 One Critical Number: Capability 7.1

Before starting, remember this number: the BI-V150's CUDA compute capability is **7.1**.

What is compute capability? Think of it as a GPU's "generation version number" — higher means newer instruction set support:

```
7.0 = Volta architecture (2017, V100)
7.1 = BI-V150's generation (improved Volta)
7.5 = Turing architecture (minimum required for INT8 quantized matrix multiply kernels)
8.0 = Ampere architecture (A100)
```

This number affects inference loading later — some parts of vLLM's quantized inference code check whether the chip reaches 7.5. The BI-V150's 7.1 does not meet that requirement. This is determined by the hardware architecture and cannot be changed through software upgrades (see Appendix B.1).

### 0.4 Five Layers You Will Encounter

This lab will let you experience these five distinct levels firsthand (each level does NOT imply the next):

1. The quantization tool can generate an output
2. `validate` passes
3. The inference framework recognizes the quantization format
4. The target chip has the corresponding optimized kernel
5. Inference accuracy and performance meet requirements

The key point this manual emphasizes: **passing earlier layers does not guarantee later layers will pass.** This experiment lets you see exactly where each layer gets blocked.

---

## Chapter 1: Quantization Principles (Beginner's Guide)

> You don't need to memorize this chapter. Read it once to build intuition, then refer back when you encounter unfamiliar parameters during the experiment.

### 1.1 Why Quantize?

A model's weights (think of these as everything the model has "memorized") are stored in **BF16** format by default, with each number occupying 2 bytes. A 35B-parameter model therefore requires about 70 GB of GPU memory.

The idea behind quantization is simple: **use a smaller number format, store weights with fewer bits**, and make the model "leaner."

### 1.2 An Intuitive Analogy

Suppose you want to record a set of temperature readings: 23.47°C, 18.92°C, 31.05°C…

- **BF16 approach**: Record precise decimals, 2 bytes per number.
- **INT8 quantization approach**: Find the range of the data (say 0–50°C), map each value to an integer from 0–255 (1 byte each), and store a "conversion factor" (scale). When needed, multiply integer × scale to recover an approximation of the original value.

Precision is slightly reduced — 23.47°C might become 23.5°C — but storage is halved.

```
original ≈ integer × scale
              ↑          ↑
         1 byte only   stored as float, shared across a group of weights
```

### 1.3 The Two Quantization Schemes in This Lab

| Scheme | Weights | Activations | Compression | Notes |
|---|---|---|---|---|
| **W8A16** | INT8 | 16-bit float | ~50% | Weights compressed; dequantized to float at inference time |
| **W8A8** | INT8 | Dynamic per-token INT8 | ~50% | Both weights and activations quantized; theoretically fastest |

Both schemes approximately halve the weight size (67G → ~36G). The difference lies in how activations are handled and what inference kernels are required.

### 1.4 Group Quantization vs Channel Quantization

This is about how large a range one scale value covers:

```
A weight matrix (think of it as a table):

Channel strategy: one scale per row
┌──────────────────────────┐
│ ● ● ● ● ● ● ● ● │← scale₁
│ ● ● ● ● ● ● ● ● │← scale₂
│ ● ● ● ● ● ● ● ● │← scale₃
└──────────────────────────┘

Group strategy (group_size=4): one scale per 4 values
┌────┬────┬────┬────┬────┐
│●●●●│●●●●│●●●●│●●●●│●●●●│
│ s₁ │ s₂ │ s₃ │ s₄ │ s₅ │← different scales for different segments in one row
└────┴────┴────┴────┴────┘
```

- **Group quantization** (used by W8A16): finer granularity, each small block has its own scale, higher precision
- **Channel quantization** (used by W8A8): coarser granularity, one scale per row, simpler to implement in kernels

### 1.5 MSE Search (The Quantization Method Used Here)

The scale selection method is called `mse_grid_search`: search across many candidate scale values and pick the one that minimizes the difference between the original and quantized weights.

```
Candidate scales: 0.001  0.002  0.003  0.004  ...
Corresponding error: large  med   small  med   ...
                                    ↑
                               pick this one!
```

This achieves higher accuracy than simply using the data's min/max to calculate the scale, at the cost of being slower — the W8A16 quantization in this lab takes about 16 minutes.

### 1.6 The Special Nature of MoE Models

Qwen3.6-35B-A3B is a **MoE (Mixture of Experts)** model:

```
Standard model: uses all parameters for every computation
   input → [all 35B parameters] → output

MoE model: has 256 "experts", only 8 are activated per token
   input → router (decides which 8 experts to activate)
                    ↓
         [expert 1][expert 2]...[expert 8] (the other 248 sleep)
                    ↓
                  output
```

These experts' weights are stored "fused" together as a 3D tensor with shape `[256 experts, output_dim, input_dim]`. FlagOS-Compressor recognizes this special layout and quantizes the entire expert bank as a unit — you cannot quantize only a subset of experts.

---

## Chapter 2: Lab Setup

### 2.1 Record Software Versions (Required for Reproducibility)

You must record the version of each component in your lab report, otherwise results cannot be reproduced. Enter each command:

```bash
# Install FlagOS-Compressor
git clone https://gitcode.com/flagos-ai/FlagOS-Compressor
cd /root/FlagOS-Compressor
pip install -e . --no-build-isolation

# FlagOS-Compressor version
cd /root/FlagOS-Compressor && git rev-parse HEAD
```

Expected output — a commit ID string, for example:

```
c19c29b069b3c5e7d940209d4534f5b37b6286db
```

Record it.

```bash
# Other component versions
python3 -c "import vllm; print('vllm', vllm.__version__)"
python3 -c "import torch; print('torch', torch.__version__)"
pip show compressed-tensors | grep Version
pip show transformers | grep Version
python3 -c "import triton; print('triton', triton.__version__)"
```

Fill all version numbers into the version table in Chapter 7.

### 2.2 Create the Output Directory

Quantization outputs will be written here:

```bash
mkdir -p /dev/shm/flagos-output
```

> **Why `/dev/shm`?**
> This is an in-memory filesystem (tmpfs) — extremely fast reads/writes, with enough capacity (252G). The downside is it disappears on reboot, but it is sufficient for this lab.

### 2.3 Confirm the Model Is Available

```bash
ls /models/Qwen3.6-35B-A3B/ | head -5
```

You should see `config.json`, `model-00001-of-00026.safetensors`, and other files.

```bash
du -sh /models/Qwen3.6-35B-A3B/
```

Should show `67G` — the BF16 source model size.

---

## Chapter 3: Inspect Model Structure

### 3.1 Run inspect

```bash
flagos-compressor inspect --input /models/Qwen3.6-35B-A3B/
```

Expected output:

```
Model: /models/Qwen3.6-35B-A3B/
Selectable weight groups
  moe: 205
  moe.routed: 82
  moe.shared: 123
  attention: 74
  mlp: 0
  linear: 279
Input storage formats
  ...
```

### 3.2 Reflection Questions (Fill Into Your Lab Report)

Check your output against the expected values:

| Item | Expected | Your observation |
|---|---|---|
| `linear` total | 279 | |
| `moe.routed` | 82 (routed experts) | |
| `moe.shared` | 123 (shared experts) | |
| `attention` | 74 | |
| `mlp` | 0 (MLP is inside expert layers in a MoE model — expected) | |

**Think about it**: Why is `linear = attention + mlp + moe`? Because `linear` is the sum of all identified linear layers. In a MoE model, MLP layers are split into expert layers, so `mlp=0` while `moe` has a count.

Save the full result as JSON for your records:

```bash
flagos-compressor inspect \
  --input /models/Qwen3.6-35B-A3B/ \
  --json > /dev/shm/flagos-output/inspect.json
```

---

## Chapter 4: Case M-W8A16 — Routed Experts + Attention, W8A16 group-128

This is the more conservative of the two schemes: quantize only routed experts and attention layers, using INT8 with a group strategy.

### 4.1 What Is a Selector?

FlagOS-Compressor uses selectors to decide which layers to quantize:

| Selector | Coverage |
|---|---|
| `linear` | All identified linear layers (attention + mlp + moe) |
| `attention` | Attention projection layers (Q/K/V/O) |
| `moe.routed` | Routed experts (gate/up/down for all 256 experts) |
| `moe.shared` | Shared experts |

This case selects `moe.routed + attention`: expert and attention layers are quantized; everything else (vision encoder, etc.) stays in BF16.

### 4.2 Step 1: Run dry-run First (No Files Written)

A dry-run is a "rehearsal" before quantization — it only computes the plan without writing any weights. **Always run a dry-run before every real quantization run.**

```bash
flagos-compressor quantize \
  --input /models/Qwen3.6-35B-A3B/ \
  --output /dev/shm/flagos-output/qwen3.6-moe-routed-attn-w8a16-g128 \
  --select moe.routed \
  --select attention \
  --bits 8 \
  --strategy group \
  --group-size 128 \
  --backend cuda \
  --dry-run
```

Expected output:

```
Execution plan
  quantization: W8A16 group mse_grid_search group_size=128
  input formats
    bf16: 156
  output formats
    compressed_tensors_int8_groupwise: 74
    compressed_tensors_int8_moe_fused: 82
  keep: 889
  unmatched quantized: 0
```

**Check each item** (tick when confirmed):

- [ ] `quantization` shows W8A16, strategy=group, group_size=128
- [ ] `unmatched quantized: 0` (no weights with unrecognized formats)
- [ ] Both selectors matched: `int8_groupwise=74` (attention) and `int8_moe_fused=82` (routed experts)
- [ ] No errors

> **Why does `unmatched quantized: 0` matter?**
> If this is not 0, the model contains weights in a format the tool cannot recognize (possibly FP4/FP8). Proceeding would produce incorrect results. This is the safety gate before quantization.

### 4.3 Step 2: Run the Real Quantization (Remove --dry-run)

Once the dry-run passes all checks, remove `--dry-run` and run for real. Since quantization takes about 15 minutes, use `nohup` to run it in the background:

```bash
nohup flagos-compressor quantize \
  --input /models/Qwen3.6-35B-A3B/ \
  --output /dev/shm/flagos-output/qwen3.6-moe-routed-attn-w8a16-g128 \
  --select moe.routed \
  --select attention \
  --bits 8 \
  --strategy group \
  --group-size 128 \
  --backend cuda \
  > /dev/shm/flagos-output/quantize-w8a16.log 2>&1 &
echo "Quantization PID: $!"
```

Monitor progress (updates every ~40 seconds):

```bash
tail -f /dev/shm/flagos-output/quantize-w8a16.log
```

You will see output like:

```
Converting:  38%|███▊      | 10/26 [06:28<10:48, 40.53s/it]
```

This means 10 of the 26 model shards have been processed, ~40 seconds each, ~16 minutes total.

Check GPU activity in a separate terminal:

```bash
ixsmi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

GPU 0 should show 100% utilization and ~1100 MiB memory usage during quantization.

When quantization completes, the log tail will show:

```
INFO: Done.
INFO: Strategy: group
INFO: W8A16 tensors: 74
INFO: W8A16 fused MoE banks: 82
INFO: Converted tensors: 156
INFO: Kept tensors: 889
INFO: Manifest: ...
INFO: Report: ...
```

**Record in your lab report**:

| Metric | Your observation |
|---|---|
| Quantization time (seconds) | |
| Any CPU fallback warnings? | |

> **What is CPU fallback?** If a quantization operator fails to run on the GPU, the tool falls back to CPU execution — much slower. A fallback is a warning sign and may affect accuracy. This lab expects 0 CPU fallback.

### 4.4 Step 3: Validate the Output

After quantization, immediately validate the output:

```bash
flagos-compressor validate \
  --input /dev/shm/flagos-output/qwen3.6-moe-routed-attn-w8a16-g128
```

**The first line must be `Valid`.** If not, stop and record the error.

Then view the quantization report:

```bash
cat /dev/shm/flagos-output/qwen3.6-moe-routed-attn-w8a16-g128/quantization_report.json | python3 -c "
import json,sys
r=json.load(sys.stdin)
print('backend:', r['backend'])
print('elapsed (s):', round(r['elapsed_s'],1))
print('quantized weights:', r['converted'])
print('kept weights:', r['kept'])
for k,v in r['ops'].items():
    print(f'op {k}: count={v[\"count\"]}, fallback={v.get(\"fallback\",0)}')
"
```

> **You should see**: backend=cuda, all operators with fallback=0 — meaning the entire quantization ran on GPU with no CPU fallback.

Save the validation result for your records:

```bash
flagos-compressor validate \
  --input /dev/shm/flagos-output/qwen3.6-moe-routed-attn-w8a16-g128 \
  --json > /dev/shm/flagos-output/validate-w8a16.json
```

### 4.5 Step 4: Compare Model Sizes

```bash
du -sh /models/Qwen3.6-35B-A3B/ \
        /dev/shm/flagos-output/qwen3.6-moe-routed-attn-w8a16-g128/
```

**Fill into your lab report**:

| Model | Size | Compression ratio |
|---|---|---|
| BF16 source | 67G | baseline |
| W8A16 output | __ G | __ % |

> **Think about it**: W8A16 only selected `moe.routed + attention` (156 weights), yet the output is only about 44% smaller rather than 50%. Why?
> Because 889 weights remain in BF16 (vision encoder, shared experts, lm_head, etc.). The unquantized portions keep the overall compression ratio below 50%.

---

## Chapter 5: Case M-W8A8 — All Linear Layers, W8A8 channel

The second scheme: quantize all 279 linear layers using INT8 with a channel strategy, and also quantize activations to dynamic INT8.

### 5.1 How Does W8A8 Differ From W8A16?

```
W8A16: Weights (W) → INT8,  Activations (A) → kept as 16-bit float
       At inference: INT8 weights × float activations

W8A8:  Weights (W) → INT8,  Activations (A) → dynamic per-token INT8
       At inference: INT8 weights × INT8 activations (requires INT8 matrix multiply kernel on the chip)
```

W8A8's potential advantage is faster inference, but it requires the chip to have the corresponding INT8 matmul kernel.

**Key W8A8 constraints to remember**:
- Must use `--strategy channel` — group strategy is not supported
- **Do NOT add `--group-size`** (W8A8 does not support group strategy)
- `--activation-bits 8` is required and fixed

### 5.2 Step 1: dry-run

```bash
flagos-compressor quantize \
  --input /models/Qwen3.6-35B-A3B/ \
  --output /dev/shm/flagos-output/qwen3.6-moe-w8a8 \
  --select linear \
  --bits 8 \
  --activation-bits 8 \
  --strategy channel \
  --backend cuda \
  --dry-run
```

Expected output:

```
Execution plan
  quantization: W8A8 channel mse_grid_search
  input formats
    bf16: 279
  output formats
    compressed_tensors_w8a8_channelwise: 197
    compressed_tensors_w8a8_channelwise_moe_fused: 82
  keep: 766
  unmatched quantized: 0
```

Note: this time `linear` (all 279 layers) is selected — 123 more than W8A16's 156, covering shared experts and vision layers.

### 5.3 Step 2: Run the Real Quantization

```bash
nohup flagos-compressor quantize \
  --input /models/Qwen3.6-35B-A3B/ \
  --output /dev/shm/flagos-output/qwen3.6-moe-w8a8 \
  --select linear \
  --bits 8 \
  --activation-bits 8 \
  --strategy channel \
  --backend cuda \
  > /dev/shm/flagos-output/quantize-w8a8.log 2>&1 &
echo "W8A8 quantization PID: $!"
```

Monitor progress:

```bash
tail -f /dev/shm/flagos-output/quantize-w8a8.log
```

W8A8 channel strategy is faster than W8A16 group (coarser granularity), estimated ~15 minutes.

> **Note**: The `tail -f` progress bar is updated by overwriting the same line. If the display appears frozen, check actual progress with:
> ```bash
> ls /dev/shm/flagos-output/qwen3.6-moe-w8a8/*.safetensors 2>/dev/null | wc -l
> ```
> Count reaching 26 AND the manifest file appearing means completion.

When done, the log will show:

```
INFO: W8A8 tensors: 197
INFO: W8A8 fused MoE banks: 82
INFO: Converted tensors: 279
```

**Important**: confirm completion using the manifest file, not just the shard count:

```bash
ls /dev/shm/flagos-output/qwen3.6-moe-w8a8/quantization_manifest.json
```

If the manifest does not appear after 26 shards are written, the process likely crashed at the final write stage. Clear the directory and retry:

```bash
rm -rf /dev/shm/flagos-output/qwen3.6-moe-w8a8
# then re-run the nohup command above
```

### 5.4 Step 3: Validate and Compare Sizes

```bash
flagos-compressor validate \
  --input /dev/shm/flagos-output/qwen3.6-moe-w8a8
```

```bash
flagos-compressor validate \
  --input /dev/shm/flagos-output/qwen3.6-moe-w8a8 \
  --json > /dev/shm/flagos-output/validate-w8a8.json

cat /dev/shm/flagos-output/qwen3.6-moe-w8a8/quantization_report.json | python3 -c "
import json,sys
r=json.load(sys.stdin)
print('backend:', r['backend'])
print('elapsed (s):', round(r['elapsed_s'],1))
print('quantized weights:', r['converted'])
for k,v in r['ops'].items():
    print(f'op {k}: count={v[\"count\"]}, fallback={v.get(\"fallback\",0)}')
"
```

```bash
du -sh /models/Qwen3.6-35B-A3B/ \
        /dev/shm/flagos-output/qwen3.6-moe-routed-attn-w8a16-g128/ \
        /dev/shm/flagos-output/qwen3.6-moe-w8a8/
```

**Fill into your lab report**:

| Model | Size | Compression | Quantized weights | CPU fallback |
|---|---|---|---|---|
| BF16 source | 67G | baseline | 0 | — |
| W8A16 (moe.routed+attention) | __ G | __ % | 156 | __ |
| W8A8 (all linear) | __ G | __ % | 279 | __ |

> **Think about it**: W8A8 quantized more layers (279 vs 156), but the output size is similar to W8A16. Why?
> The INT8 weight storage is the same size regardless of scheme. The difference is in scale count and format. W8A8 uses channel strategy (fewer scales), W8A16 uses group-128 (more scales), but the scale storage overhead doesn't significantly affect total size.

---

## Chapter 6: Attempt Inference Loading (Observe the Framework's Response)

> This chapter may not succeed — but every error message is itself a learning outcome. Inference failure ≠ experiment failure.

### 6.1 Can Quantized Outputs Be Used for Inference Directly?

Revisit the five layers from §0.4. Completing quantization + passing validate only means:

✅ FlagOS-Compressor generated a correctly formatted output

It does **not** guarantee: the inference framework can load it / the chip has the required kernel / accuracy is acceptable.

Now let's empirically test the inference framework's response.

### 6.2 Attempt to Load the W8A16 Output

In a terminal, start the vLLM inference service:

```bash
export VLLM_PLUGINS=fl
export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup vllm serve \
  /dev/shm/flagos-output/qwen3.6-moe-routed-attn-w8a16-g128/ \
  -tp 4 \
  --served-model-name qwen-w8a16 \
  --port 8012 \
  --enforce-eager \
  --max-model-len 4096 \
  > /dev/shm/flagos-output/vllm-w8a16.log 2>&1 &
echo "vLLM PID: $!"
```

Monitor the log:

```bash
tail -f /dev/shm/flagos-output/vllm-w8a16.log
```

**Key message you will likely see**:

```
RuntimeError: Quantization scheme is not supported for the current GPU.
Min capability: 75. Current capability: 71.
```

This means vLLM's compressed-tensors plugin checked the chip's compute capability when loading W8A16 — it requires at least 7.5, but the BI-V150 is 7.1, so it refuses to load.

This is the real-world impact of the capability 7.1 discussed in §0.3 — not an abstract concept, but a concrete error you just witnessed.

```bash
# Stop the service and free GPU memory
kill $(ps aux | grep "vllm serve" | grep -v grep | awk '{print $1}') 2>/dev/null; true
```

### 6.3 Attempt to Load the W8A8 Output

```bash
export VLLM_PLUGINS=fl
export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup vllm serve \
  /dev/shm/flagos-output/qwen3.6-moe-w8a8/ \
  -tp 4 \
  --served-model-name qwen-w8a8 \
  --port 8013 \
  --enforce-eager \
  --max-model-len 4096 \
  > /dev/shm/flagos-output/vllm-w8a8.log 2>&1 &
echo "vLLM PID: $!"
```

```bash
tail -f /dev/shm/flagos-output/vllm-w8a8.log
```

Watch the log and note these keywords:

| Keyword | Meaning |
|---|---|
| `compressed-tensors` | vLLM recognized the quantization format |
| `Min capability` / `Current capability` | Capability check result |
| `WorkerProc initialization failed` | Worker process failed to initialize |
| `Application startup complete` | Service is ready (if this appears, loading succeeded) |

Record what you observe in the Chapter 7 lab report.

```bash
kill $(ps aux | grep "vllm serve" | grep -v grep | awk '{print $1}') 2>/dev/null; true
```

### 6.4 What Inference Failure Means

Inference loading failure does not mean this experiment failed. The result itself is what the experiment set out to observe:

- **FlagOS-Compressor worked perfectly**: both schemes completed quantization, validate passed, 0 CPU fallback
- **Inference failures occur at different layers**:
  - W8A16 fails at layer 3 (inference framework loading): capability check directly rejects it
  - W8A8 fails between layers 3–4: can enter initialization, but crashes in the WorkerProc stage
- **This is a live demonstration of the five-layer gap**: the quantization tool and inference framework are two independent systems, each with their own limitations

Once the chip vendor fixes the missing kernels in vllm-plugin-FL, the same quantized outputs can be used for inference directly — no re-quantization needed.

---

## Chapter 7: Lab Report Template

Copy the template below into your report and fill it in:

```
Name:
Date:
Lab environment IP (hostname -I):
FlagOS-Compressor commit:
vLLM version:
PyTorch version:

═══════════ Chapter 3: inspect ═══════════
linear total:
moe.routed count:
moe.shared count:
attention count:
Source weight format (BF16 / FP8 / FP4):

═══════════ Chapter 4: W8A16 ═══════════
dry-run unmatched quantized:
int8_groupwise (attention) matched:
int8_moe_fused (routed experts) matched:
Quantization time (seconds):
Any CPU fallback during quantization:
validate result (Valid / error):
Output size:              Compression ratio:

═══════════ Chapter 5: W8A8 ═══════════
dry-run unmatched quantized:
Total quantized weights (converted):
Quantization time (seconds):
Any CPU fallback during quantization:
validate result (Valid / error):
Output size:              Compression ratio:

═══════════ Comparison ═══════════
│ Metric                │ W8A16     │ W8A8      │
│ Output size           │           │           │
│ Quantized weights     │ 156       │ 279       │
│ Time (seconds)        │           │           │
│ CPU fallback          │           │           │
│ validate              │           │           │

═══════════ Chapter 6: Inference Loading ═══════════
W8A16 loading result:
  - Error message (if any):
  - Did vLLM recognize compressed-tensors:
W8A8 loading result:
  - Error message (if any):
  - Did vLLM recognize compressed-tensors:

═══════════ Conclusion (use this format) ═══════════
Using FlagOS-Compressor <commit>, model Qwen3.6-35B-A3B (BF16):

W8A16 case: compressed-tensors output generated; inspect, dry-run, and validate passed.
On Iluvatar BI-V150 ×4, vllm-plugin-FL <commit>, PyTorch <version>,
vLLM loading result: <success / failure, reason: ___>.

W8A8 case: compressed-tensors output generated; inspect, dry-run, and validate passed.
Under the same environment,
vLLM loading result: <success / failure, reason: ___>.

This conclusion applies only to the above model, quantization recipe, and software/hardware version combination.
```

---

## Appendix A: Common Errors and Fixes

| Symptom | Cause | Fix |
|---|---|---|
| `flagos-compressor: command not found` | Entry script not in PATH | Use `python3 -m flagos_compressor.cli.main` instead |
| `No weights selected` | No `--select` provided or typo | Check inspect output, confirm selector name |
| `selectors did not match any supported tensors` | Selector matched nothing | Run inspect first, confirm that selector has a count |
| `unmatched quantized` is not 0 | Weights in unrecognized low-precision format | Record the error; see FlagOS-Compressor manual §12 |
| `in_features must be divisible by group_size` | A selected layer's in_features is not divisible by group_size | Try a smaller group_size (64 or 32) |
| `group_size must be omitted for channel strategy` | `--group-size` passed with W8A8 | Remove the `--group-size` parameter |
| `validate` returns `Valid` but inference fails | Inference framework kernel doesn't support this scheme | See §6.4 — this is a normal experimental outcome |
| `Min capability: 75. Current capability: 71.` | BI-V150 capability below the requirement | See §0.3 — this is a hardware limitation |
| 26 shards written but no `quantization_manifest.json` | Process crashed at the final write stage | Clear the output directory and retry; restart container if it recurs |

---

## Appendix B: Supplementary Theory

### B.1 Why Can't Capability 7.1 Be Upgraded to 7.5?

CUDA compute capability is an intrinsic property of the hardware architecture, determined by the chip's physical design:

- 7.5 = Turing architecture: physically has INT8 Tensor Core units on-chip
- 7.1 = BI-V150: does not have these units

It is like asking a phone with no camera to take photos through a software update — missing hardware cannot be supplied by drivers or firmware.

**What is Triton fallback?** PR#336 added a W8A8 kernel written in Triton to vLLM. It does not depend on INT8 Tensor Cores — it performs integer operations using general-purpose GPU compute units, and in principle can run on 7.1. However, it still requires additional fixes at the vllm-plugin-FL layer to work completely. This is why W8A8 inference fails: PR#336's Triton kernel needs more accompanying vllm-plugin-FL changes to function end-to-end.

### B.2 Quantization Only Reduces Weight Memory

A common beginner misconception: GPU memory usage decreases proportionally after quantization. In practice:

- Weight memory: reduced in proportion to quantization ratio (W8A16 weights ~halved)
- KV cache: unaffected
- Activations and intermediate tensors: unaffected for W8A16; for W8A8, activation quantization happens dynamically at inference time
- Framework overhead, communication buffers: unaffected

The actual memory saving after quantization is therefore smaller than the weight compression ratio alone suggests.
