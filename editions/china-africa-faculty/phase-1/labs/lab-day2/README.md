# CANN-Triton Lab Day 2

Hands-on lab for the international training course
**"Software and Hardware Foundations of Intelligent Computing Systems"**.

Day 2 picks up where Day 1 left off: the tiled Triton matmul in
`04_matmul_compare.py` runs correctly but is 3-5x slower than
`torch.matmul`. Today we close most of that gap -- **not** by learning
more Triton syntax, but by building an empirical model of the Ascend
Cube compute path and tuning our kernel against measured hardware
facts.

---

## Quick Start

Each new terminal:

```bash
source setup.sh
```

Then run the four steps in order:

```bash
python 00_baseline.py              # calibrate measurement
python 01_buffer_model.py          # main event: measure Cube buffers
python 02_autotune_roofline.py     # autotune + BK sweep
python 03_summary.py               # final comparison table
```

---

## What You'll Do

| File                        | What you do                                        | Time    |
|-----------------------------|-----------------------------------------------------|---------|
| `00_baseline.py`            | Run only. Record starting numbers.                  | ~10 min |
| `01_buffer_model.py`        | Observe failures, fill in two capacity values.      | ~35 min |
| `02_autotune_roofline.py`   | Fill in the legal-tile config generator.            | ~40 min |
| `03_summary.py`             | Run only. Read the summary table.                   | ~15 min |

Total ~100 minutes of hands-on, leaving time for discussion.

---

## The Story

**Day 1 left us asking**: why is our straightforward tiled matmul so
much slower than `torch.matmul`? Is the gap a fact of life ("vendors
always win") or can we reason about it?

**Day 2 answers**: it is reasonable, and it is closable. We'll make
and measure three claims about the Ascend Cube:

1. The Cube compute path is a **multi-stage pipeline** with several
   independent on-chip buffers, not a single shared scratchpad.
2. Each stage has a **concrete, measurable capacity** that bounds what
   tile shapes we can legally pick.
3. Inside the legal space, **which tile is best is not obvious** --
   the optimum is asymmetric and depends on BK alignment in a way the
   docs don't fully capture. We let autotune find it, and we build
   roofline intuition for why.

By the end, our Triton matmul at fp32 2048^3 should be close to or
matching the hand-tuned CANN vendor kernel.

The detailed narrative, with the "hypothesis → experiment → data"
structure for each step, is in [`handout.md`](./handout.md).

---

## Environment

Identical to Day 1:

| Component       | Version           |
|-----------------|-------------------|
| Hardware        | Ascend 910C (1 card, 2 dies) |
| CPU             | aarch64 (Kunpeng / Phytium) |
| CANN            | 8.3.RC2           |
| PyTorch         | 2.8.0 (CPU build) |
| torch_npu       | 2.8.0             |
| Triton-Ascend   | 3.2.0             |
| Python          | 3.11              |

If `source setup.sh` prints all five components with versions, you are
ready.

---

## Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: ... NPU not found` | Forgot `source setup.sh` in this terminal | Re-source it |
| Step 02 takes 1-3 min on first run | autotune compiles every config | Wait. Subsequent runs use cache |
| Large MLIR traceback on step 01 Part A | Expected -- some configs fail on purpose | Our `try_config` catches it; read the compact row at the end |
| Numbers fluctuate run-to-run | Small shape + few iters = noise | Stick to the script's shapes; `bench_ms` already warms up |

To clear the Triton compilation cache (e.g. after editing a kernel):

```bash
rm -rf ~/.triton/cache ~/.triton/dump
```

---

## Resources

- Day 1 lab: https://gitee.com/jieran-zhang/lab_day1
- Triton-Ascend programming guide: https://ascend.github.io/triton-ascend/sources/programming-guide/introduction.html
- Triton-Ascend migration guide (GPU → Ascend): https://github.com/triton-lang/triton-ascend/blob/main/docs/en/migration_guide/migrate_from_gpu.md

---

## License

CC BY-NC 4.0 — Creative Commons Attribution-NonCommercial 4.0 International.
