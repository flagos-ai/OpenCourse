#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlagPrism / FlagTree debugger report summary + visualization tool

Usage (run on an Iluvatar machine):
    # Summarize the latest report (auto-finds the newest *.json in the dir),
    # print the text table
    python3 flagprism_report_summary.py

    # Specify a particular json report
    python3 flagprism_report_summary.py /tmp/flagtree_debugger_nan/xxx_run1.json

    # Specify a directory (by default all flagtree_debugger_* dirs under /tmp
    # are scanned)
    python3 flagprism_report_summary.py --dir /tmp/flagtree_debugger_nan

    # Also output a chart (requires matplotlib; skipped automatically if not
    # installed, text only)
    python3 flagprism_report_summary.py --plot

    # Save the chart to a specified path (default: *_summary.png in the report's
    # own directory)
    python3 flagprism_report_summary.py --plot --out /tmp/summary.png

Design notes:
- The field hierarchy of the report json may differ slightly across FlagPrism
  versions, so this does "adaptive probing": it recursively searches each
  statement record for nan_count/inf_count/zero_count/mean/min/max/l2_norm/
  element_count.
- If field names don't match, use --dump to print the structure it actually
  parsed out, and send the output to me so I can fix it accordingly.
"""

import argparse
import glob
import json
import os
import sys

# Stat fields of interest -> aliases that may appear in the report
# (matched lowercase)
STAT_KEYS = {
    "element_count": ["element_count", "elem_count", "count", "numel"],
    "nan_count":     ["nan_count", "nans", "num_nan"],
    "inf_count":     ["inf_count", "infs", "num_inf"],
    "zero_count":    ["zero_count", "zeros", "num_zero"],
    "mean":          ["mean", "avg"],
    "min":           ["min", "minimum"],
    "max":           ["max", "maximum"],
    "l2_norm":       ["l2_norm", "l2", "norm", "l2norm"],
}


def _coerce_number(v):
    """Extract a number from things like [16 (U64)] / "16" / 16 / {'value': 16}."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, dict):
        for k in ("value", "val", "data"):
            if k in v:
                return _coerce_number(v[k])
        return None
    if isinstance(v, list):
        return _coerce_number(v[0]) if v else None
    if isinstance(v, str):
        s = v.strip()
        # Forms like "16 (U64)" / "-4 (F32)" / "nan"
        token = s.split()[0] if s else ""
        token = token.strip("[]")
        try:
            return float(token)
        except ValueError:
            if token.lower() == "nan":
                return float("nan")
            if token.lower() in ("inf", "+inf"):
                return float("inf")
            if token.lower() == "-inf":
                return float("-inf")
            return None
    return None


def _find_stat(container, aliases):
    """Find a field by alias in a dict/nested structure, return the extracted number."""
    if not isinstance(container, dict):
        return None
    lower = {str(k).lower(): k for k in container.keys()}
    for a in aliases:
        if a in lower:
            return _coerce_number(container[lower[a]])
    return None


def _extract_summary_from_node(node):
    """
    Given a node that may contain a summary, return {stat_name: number}.
    Prefers the node's 'summary'; if not found, searches the node itself.
    """
    search_spaces = []
    if isinstance(node, dict):
        if isinstance(node.get("summary"), dict):
            search_spaces.append(node["summary"])
        search_spaces.append(node)
    stats = {}
    for name, aliases in STAT_KEYS.items():
        for space in search_spaces:
            val = _find_stat(space, aliases)
            if val is not None:
                stats[name] = val
                break
    return stats


def _walk_statements(obj, path=""):
    """
    Recursively traverse the json, yielding (loc/statement text, stats dict).
    Recognition rule: any dict that (itself or its result child node) contains
    stat fields like nan_count/min is treated as a "statement record".
    """
    results = []

    def visit(node, label):
        if isinstance(node, dict):
            # The readable label of this record: prefer statement / source_loc / name
            stmt = (node.get("statement") or node.get("source_loc")
                    or node.get("loc") or node.get("name") or label)
            # Stats may hang on the node itself, or on a result child node
            stats = _extract_summary_from_node(node)
            # Common: {'[result x]': {'summary': {...}}} or 'result'
            if not stats:
                for k, v in node.items():
                    if isinstance(v, dict) and ("result" in str(k).lower()
                                                or str(k).lower() == "result"):
                        stats = _extract_summary_from_node(v)
                        if stats:
                            break
            if stats:
                results.append((str(stmt), stats))
            for k, v in node.items():
                visit(v, f"{label}.{k}" if label else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                visit(v, f"{label}[{i}]")

    visit(obj, path)
    # Deduplicate (keep the first when the same statement hits multiple times)
    seen, deduped = set(), []
    for stmt, stats in results:
        key = (stmt, tuple(sorted(stats.items())))
        if key not in seen:
            seen.add(key)
            deduped.append((stmt, stats))
    return deduped


def _fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        if v != v:
            return "nan"
        if v in (float("inf"), float("-inf")):
            return "inf" if v > 0 else "-inf"
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.{nd}g}"
    return str(v)


def print_text_summary(records, title=""):
    if title:
        print(f"\n=== {title} ===")
    if not records:
        print("(No statements with stat fields were parsed; use --dump to see the raw structure)")
        return

    cols = ["#", "statement", "elem", "nan", "inf", "zero", "min", "max", "mean", "l2"]
    rows = []
    for i, (stmt, s) in enumerate(records):
        short = stmt if len(stmt) <= 48 else stmt[:45] + "..."
        rows.append([
            str(i),
            short,
            _fmt(s.get("element_count")),
            _fmt(s.get("nan_count")),
            _fmt(s.get("inf_count")),
            _fmt(s.get("zero_count")),
            _fmt(s.get("min")),
            _fmt(s.get("max")),
            _fmt(s.get("mean")),
            _fmt(s.get("l2_norm")),
        ])

    widths = [max(len(cols[c]), max((len(r[c]) for r in rows), default=0))
              for c in range(len(cols))]
    line = "  ".join(cols[c].ljust(widths[c]) for c in range(len(cols)))
    print(line)
    print("  ".join("-" * widths[c] for c in range(len(cols))))
    for r in rows:
        print("  ".join(r[c].ljust(widths[c]) for c in range(len(cols))))

    # Conclusion: the first statement to show inf / nan
    first_inf = next((i for i, (_, s) in enumerate(records)
                      if (s.get("inf_count") or 0) > 0), None)
    first_nan = next((i for i, (_, s) in enumerate(records)
                      if (s.get("nan_count") or 0) > 0), None)
    print()
    if first_inf is not None:
        print(f">> First statement showing inf: #{first_inf}  {records[first_inf][0]}  "
              f"(inf={_fmt(records[first_inf][1].get('inf_count'))})")
    if first_nan is not None:
        print(f">> First statement showing nan: #{first_nan}  {records[first_nan][0]}  "
              f"(nan={_fmt(records[first_nan][1].get('nan_count'))})")
    if first_inf is None and first_nan is None:
        print(">> No nan / inf found, values look clean.")


def make_plot(records, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")  # works even in a headless environment
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] Skipping chart output: matplotlib unavailable ({e})")
        print("[plot] Install: pip install matplotlib")
        return False

    n = len(records)
    if n == 0:
        print("[plot] No data, skipping.")
        return False

    idx = list(range(n))
    labels = [f"#{i}\n{(stmt.split(chr(10))[0])[:18]}" for i, (stmt, _) in enumerate(records)]
    nan = [records[i][1].get("nan_count") or 0 for i in idx]
    inf = [records[i][1].get("inf_count") or 0 for i in idx]
    zero = [records[i][1].get("zero_count") or 0 for i in idx]

    def safe(seq):
        return [(x if isinstance(x, (int, float)) and x == x else 0) for x in seq]

    mn = safe([records[i][1].get("min") for i in idx])
    mx = safe([records[i][1].get("max") for i in idx])
    mean = safe([records[i][1].get("mean") for i in idx])
    l2 = safe([records[i][1].get("l2_norm") for i in idx])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, n * 1.2), 9))

    # Top: nan/inf/zero count bars
    w = 0.25
    ax1.bar([i - w for i in idx], nan, width=w, label="nan", color="#d62728")
    ax1.bar(idx, inf, width=w, label="inf", color="#ff7f0e")
    ax1.bar([i + w for i in idx], zero, width=w, label="zero", color="#7f7f7f")
    ax1.set_title("nan / inf / zero count per statement")
    ax1.set_xticks(idx)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Bottom: min/max/mean/l2 as statements progress
    ax2.plot(idx, mn, "o-", label="min")
    ax2.plot(idx, mx, "o-", label="max")
    ax2.plot(idx, mean, "o-", label="mean")
    ax2.plot(idx, l2, "s--", label="l2_norm", alpha=0.7)
    ax2.set_title("value range per statement (min/max/mean/l2)")
    ax2.set_xticks(idx)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"[plot] Chart saved: {out_path}")
    return True


def find_latest_json(dirs):
    candidates = []
    for d in dirs:
        candidates += glob.glob(os.path.join(d, "*.json"))
    # Exclude op_log, prefer the main report
    main = [c for c in candidates if "op_log" not in os.path.basename(c)]
    pool = main or candidates
    if not pool:
        return None
    return max(pool, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser(description="FlagPrism debugger report summary/visualization")
    ap.add_argument("json", nargs="?", help="report json path (omit to auto-find the latest)")
    ap.add_argument("--dir", action="append", default=[],
                    help="which directory to look for reports in, can be specified multiple times")
    ap.add_argument("--plot", action="store_true", help="also output a png chart")
    ap.add_argument("--out", help="chart output path (default *_summary.png in the report's dir)")
    ap.add_argument("--dump", action="store_true",
                    help="print the parsed raw structure, for troubleshooting field mismatches")
    args = ap.parse_args()

    if args.json:
        path = args.json
    else:
        search_dirs = args.dir or glob.glob("/tmp/flagtree_debugger_*") or ["."]
        path = find_latest_json(search_dirs)
        if not path:
            print("No *.json report found. Specify a path as a positional argument, or use --dir to specify a directory.")
            sys.exit(1)
        print(f"[auto] Using latest report: {path}")

    if not os.path.isfile(path):
        print(f"File does not exist: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = _walk_statements(data)

    if args.dump:
        print(json.dumps({"parsed_records": [
            {"statement": s, "stats": st} for s, st in records
        ]}, ensure_ascii=False, indent=2))

    print_text_summary(records, title=os.path.basename(path))

    if args.plot:
        out = args.out or os.path.splitext(path)[0] + "_summary.png"
        make_plot(records, out)


if __name__ == "__main__":
    main()
