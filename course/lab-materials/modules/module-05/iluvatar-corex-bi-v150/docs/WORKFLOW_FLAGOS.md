# FlagOS / BI-V150 闭环

学生仓：https://gitee.com/sunxt-0719/day2-lab5

## 与 Lab1

[Lab1](https://gitee.com/sunxt-0719/lab1) 后半部分的 GEMM 已整合进本仓库的 `kernels/gemm_triton.py`。只需克隆本仓库；来源、固定配置和原生 autotune 的对应关系见 [`GEMM_SOURCE.md`](GEMM_SOURCE.md)。

## 闭环

```text
本地编辑 → git push → FlagOS git pull → 跑 00–04
         → 写入 results/ → git push → 本地 pull
```

## 首次上机

```bash
cd labs/module-05
source env_corex.sh
bash setup.sh
python3 00_check_env.py
python3 00_smoke_cuda.py
python3 01_benchmark.py --scenario offline --repeats 3
python3 01_benchmark.py --scenario server --repeats 3
python3 02_roofline.py
python3 03_profiling.py --capture
python3 04_summary.py
```

若需 Roofline 图：`pip install matplotlib`（无图时仍有 `roofline.json`）。

操作说明以根目录 [`README.md`](../README.md) 为准。

Server SLA 使用排队加服务的响应延迟。Roofline 默认峰值仅为待核实的参考假设；实际瓶颈需计数器证据。汇总检查不通过会返回非零状态；仅探测到 profiler 工具不算完成采集。修改测量代码或重跑 benchmark 后，应重新生成后续结果。

Step 04 必须验证 ixsmi 和 ixsys 的采集结果；ixkn 为可选项，缺失、崩溃或报告校验失败只产生警告，不会单独导致验收失败。当前 Step 03 仍会尝试 ixkn，并在其失败时返回非零；若只有 ixkn 失败，单独运行 Step 04 检查已生成的结果即可。本次仅调整验收逻辑，更新后可直接复查原有结果，无需重新跑 benchmark。
