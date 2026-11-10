#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Expectation vs reality: visualize "what you should get" vs "what you actually
get" from the FlagTree Profiler Tianshu experiment. Data comes from the real
viewer output on Tianshu BI-V150, 2026-11-05.

Generates four charts:
  1. Capability coverage radar (expected vs actual)
  2. Injected metrics bytes/flops32: expected vs measured (should match exactly)
  3. Measured host_launch_duration: timing comparison of two kernels
  4. Metric availability heat table (which metrics are expected vs which --list actually shows)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["axes.unicode_minus"] = False
# Prefer a font that can display Chinese; fall back to English labels if none found
ZH = True
for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"]:
    try:
        from matplotlib.font_manager import findfont, FontProperties
        if findfont(FontProperties(family=f), fallback_to_default=False):
            plt.rcParams["font.sans-serif"] = [f]
            break
    except Exception:
        continue
else:
    ZH = False


def L(zh, en):
    return zh if ZH else en


# ---------------------------------------------------------------------------
# Chart 1: capability coverage radar
# ---------------------------------------------------------------------------
def radar():
    labels = [
        L("ixKN硬件采集", "ixKN capture"),
        L("按kernel聚合时间", "per-kernel time"),
        L("IR指令级拆解", "IR instr. level"),
        L("cpu_timed_scope", "cpu_timed_scope"),
        L("注入bytes/flops", "injected metrics"),
        L("scope树+时间合并", "scope+time merge"),
    ]
    # 1.0=fully achieved, 0.5=partial/with reservations, 0=not achieved
    expected = [1, 1, 1, 1, 1, 1]
    actual =   [1, 1, 1, 0.6, 1, 0.1]

    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    ang += ang[:1]
    exp = expected + expected[:1]
    act = actual + actual[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(ang, exp, "o-", color="#4C8BF5", lw=2, label=L("期待", "Expected"))
    ax.fill(ang, exp, color="#4C8BF5", alpha=0.12)
    ax.plot(ang, act, "o-", color="#E8743B", lw=2, label=L("实际", "Actual"))
    ax.fill(ang, act, color="#E8743B", alpha=0.20)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1.0"], fontsize=8)
    ax.set_title(L("能力覆盖:期待 vs 实际", "Capability coverage: Expected vs Actual"),
                 fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    fig.savefig("gap_1_radar.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 2: injected metrics expected vs measured (single call, should be exactly equal)
# ---------------------------------------------------------------------------
def injected():
    n = 1 << 20
    rows, cols = 2048, 1024
    items = [
        ("vector_add\nbytes", n * 4 * 3, 12582912),
        ("vector_add\nflops32", n, 1048576),
        ("rmsnorm\nbytes", rows * cols * 4 * 3, 25165824),
        ("rmsnorm\nflops32", rows * cols * 4, 8388608),
    ]
    names = [i[0] for i in items]
    exp = [i[1] for i in items]
    act = [i[2] for i in items]

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, exp, w, label=L("期待(公式)", "Expected (formula)"),
                color="#4C8BF5")
    b2 = ax.bar(x + w / 2, act, w, label=L("实测(viewer)", "Actual (viewer)"),
                color="#E8743B", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(L("数值(对数轴)", "value (log)"))
    ax.set_yscale("log")
    ax.set_title(L("注入指标:期待 vs 实测(完全吻合,差距=0)",
                   "Injected metrics: Expected vs Actual (exact match, gap=0)"),
                 fontsize=12)
    ax.legend()
    for b in list(b1) + list(b2):
        ax.annotate(f"{int(b.get_height()):,}",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=7, rotation=0)
    fig.tight_layout()
    fig.savefig("gap_2_injected.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 3: measured host_launch_duration (two kernels)
# ---------------------------------------------------------------------------
def duration():
    names = ["rmsnorm_kernel", "vector_add_kernel"]
    ms = [152740750 / 1e6, 124391333 / 1e6]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(names, ms, color=["#E8743B", "#4C8BF5"])
    ax.set_xlabel(L("host_launch_duration (毫秒, 累加值)",
                    "host_launch_duration (ms, accumulated)"))
    ax.set_title(L("实测主机侧 launch 耗时(落在 kernel 树)",
                   "Measured host launch duration (on kernel tree)"), fontsize=12)
    for b, v in zip(bars, ms):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.1f} ms",
                va="center", fontsize=10)
    ax.text(0.98, 0.04,
            L("注:vector_add 被 replay 十几次、rmsnorm 仅 1 次,\n单次 rmsnorm 实际更贵",
              "note: vector_add replayed ~10x, rmsnorm 1x;\nper-call rmsnorm is costlier"),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="gray")
    fig.tight_layout()
    fig.savefig("gap_3_duration.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 4: metric availability heat table (expected vs actual)
# ---------------------------------------------------------------------------
def availability():
    metrics = [
        L("GPU执行时间 time", "GPU time"),
        L("DRAM带宽 bandwidth", "bandwidth"),
        L("占用率 occupancy", "occupancy"),
        L("指令计数 instruction", "instruction"),
        L("主机launch耗时", "host_launch_dur"),
        L("IR指令拆解", "IR instr."),
        L("访存字节 memory_bytes", "memory_bytes"),
        L("cpu_time", "cpu_time"),
        L("注入 bytes/flops", "bytes/flops"),
    ]
    # 1=available, 0=not available
    expected = [1, 1, 1, 1, 1, 1, 1, 1, 1]   # ideal case in the docs: "want them all"
    actual =   [0, 0, 0, 0, 1, 1, 1, 1, 1]   # what Tianshu --list actually has

    data = np.array([expected, actual])
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([L("期待", "Expected"), L("天数实际", "Actual")])
    for i in range(2):
        for j in range(len(metrics)):
            ax.text(j, i, "YES" if data[i, j] else "NO",
                    ha="center", va="center",
                    color="black", fontsize=9, fontweight="bold")
    ax.set_title(L("指标可用性:期待(通用文档) vs 天数 ixKN 实际 --list",
                   "Metric availability: Expected (docs) vs Tianshu --list"),
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("gap_4_availability.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    radar()
    injected()
    duration()
    availability()
    print("Generated: gap_1_radar.png / gap_2_injected.png / "
          "gap_3_duration.png / gap_4_availability.png")
