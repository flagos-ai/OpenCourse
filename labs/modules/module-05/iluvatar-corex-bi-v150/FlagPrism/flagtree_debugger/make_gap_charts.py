#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Expectation vs reality: visualize "what you should get" vs "what you actually
get" from the FlagPrism Debugger Tianshu experiment. Data comes from the real
capture results recorded in README steps 5 and 7
(Tianshu BI-V150 / iluvatar CoreX, mainly level=1).

Generates four charts:
  1. Capability coverage radar (expected vs actual)
  2. NaN propagation chain: nan/inf/zero counts per statement of exp_catch_nan
     (shows at a glance which statement bad values first surface at)
  3. Comparison of what level 1 vs level 2 can get (fields / dumped data)
  4. Capability availability heat table (which capabilities are expected vs
     which the current Tianshu version actually has)
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
        L("数值summary采集", "value summary"),
        L("地址摘要采集", "addr summary"),
        L("level2完整落盘", "level2 dump"),
        L("逐语句定位NaN", "per-stmt locate"),
        L("逐lane地址(addr_lv2)", "per-lane addr"),
        L("数值异常自动判定", "auto diagnose"),
    ]
    # 1.0=fully achieved, 0.5=partial/with reservations, 0=not achieved
    expected = [1, 1, 1, 1, 1, 1]
    actual =   [1, 0.6, 1, 1, 0.4, 0.2]

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
# Chart 2: NaN propagation chain (real level 1 summary of exp_catch_nan)
# ---------------------------------------------------------------------------
def nan_chain():
    # Real capture results recorded in README step 5 (BLOCK_SIZE=16)
    stmts = [
        L("x = tl.load(...)", "x = load"),
        L("d = tl.load(...)", "d = load"),
        L("ratio = x / d", "ratio = x/d"),
        L("root = tl.sqrt(ratio)", "root = sqrt"),
    ]
    zero = [0, 1, 0, 0]   # d has one 0: the divide-by-zero hazard is planted here
    inf =  [0, 0, 1, 0]   # x/0 turns into inf here
    nan =  [0, 0, 0, 8]   # sqrt of a negative (plus that inf) turns into 8 nans here

    x = np.arange(len(stmts))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, zero, w, label=L("zero_count", "zero_count"), color="#7f7f7f")
    ax.bar(x,     inf,  w, label=L("inf_count", "inf_count"), color="#ff7f0e")
    ax.bar(x + w, nan,  w, label=L("nan_count", "nan_count"), color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(stmts, fontsize=9)
    ax.set_ylabel(L("计数(16 lane)", "count (of 16 lanes)"))
    ax.set_title(L("NaN 传播链:坏值逐语句累积(Debugger level 1 真实 summary)",
                   "NaN propagation: bad values accumulate per statement"),
                 fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    # Annotate where bad values surface
    ax.annotate(L("除零隐患\nzero=1", "zero=1"), (1 - w, 1), (1 - w, 3.2),
                ha="center", fontsize=8, color="#555",
                arrowprops=dict(arrowstyle="->", color="#555"))
    ax.annotate(L("除零暴露\ninf=1", "inf=1"), (2, 1), (2, 3.2),
                ha="center", fontsize=8, color="#b5651d",
                arrowprops=dict(arrowstyle="->", color="#b5651d"))
    ax.annotate(L("负数开方\nnan=8", "nan=8"), (3 + w, 8), (3 - 0.1, 6.2),
                ha="center", fontsize=8, color="#a01818",
                arrowprops=dict(arrowstyle="->", color="#a01818"))
    fig.tight_layout()
    fig.savefig("gap_2_nan_chain.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 3: what level 1 vs level 2 can get
# ---------------------------------------------------------------------------
def levels():
    items = [
        L("数值summary\n(nan/inf/mean/l2)", "value summary"),
        L("地址摘要\n(首尾/跨度/模式)", "addr summary"),
        L("完整数值\n.npy", "full value npy"),
        L("逐lane地址\n.npy", "per-lane addr npy"),
        L("定位到\n具体lane", "locate lane"),
    ]
    # 1=available, 0.5=conditional, 0=not available
    lv1 = [1, 1, 0, 0, 0]
    lv2 = [1, 1, 1, 0.5, 1]   # per-lane addr 0.5: only when the backend supports that pointer/mask pattern

    x = np.arange(len(items))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5))
    b1 = ax.bar(x - w / 2, lv1, w, label=L("level 1 (日常)", "level 1"),
                color="#4C8BF5")
    b2 = ax.bar(x + w / 2, lv2, w, label=L("level 2 (疑难)", "level 2"),
                color="#E8743B", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(items, fontsize=8.5)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels([L("无", "no"), L("带条件", "cond."), L("有", "yes")])
    ax.set_ylim(0, 1.15)
    ax.set_title(L("level 1 vs level 2 能拿到什么(报告大小:1 小 / 2 大)",
                   "What level 1 vs level 2 gives you"), fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.text(3 + w / 2, 0.52, L("仅后端支持\n该指针/mask", "backend-dep."),
            ha="center", va="bottom", fontsize=7, color="gray")
    fig.tight_layout()
    fig.savefig("gap_3_levels.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 4: capability availability heat table (expected vs current Tianshu version actual)
# ---------------------------------------------------------------------------
def availability():
    caps = [
        L("数值summary", "value summary"),
        L("地址摘要", "addr summary"),
        L("level2完整落盘", "level2 dump"),
        L("逐lane完整地址", "per-lane addr"),
        L("定位到语句", "locate stmt"),
        L("指令timeline", "instr timeline"),
        L("fp转换精度诊断", "fp cast diag"),
        L("除零/负开方自动判定", "auto zero/sqrt"),
        L("穿透任意callgraph", "callgraph ABI"),
        L("warp/异常lane聚类", "warp/lane cluster"),
    ]
    # 1=available, 0.5=partial/with reservations, 0=not available
    expected = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   # ideal case: "want them all"
    actual =   [1, 1, 1, 0.5, 1, 0, 1, 0, 0, 0]  # current Tianshu version actual

    data = np.array([expected, actual])
    fig, ax = plt.subplots(figsize=(12, 2.9))
    ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(caps)))
    ax.set_xticklabels(caps, rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([L("期待", "Expected"), L("天数实际", "Actual")])
    for i in range(2):
        for j in range(len(caps)):
            v = data[i, j]
            txt = "YES" if v == 1 else ("~" if v == 0.5 else "NO")
            ax.text(j, i, txt, ha="center", va="center",
                    color="black", fontsize=9, fontweight="bold")
    ax.set_title(L("能力可用性:期待(理想) vs 天数 CoreX 当前版本实际",
                   "Capability availability: Expected vs Tianshu CoreX actual"),
                 fontsize=12)
    fig.tight_layout()
    fig.savefig("gap_4_availability.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    radar()
    nan_chain()
    levels()
    availability()
    print("Generated: gap_1_radar.png / gap_2_nan_chain.png / "
          "gap_3_levels.png / gap_4_availability.png")
