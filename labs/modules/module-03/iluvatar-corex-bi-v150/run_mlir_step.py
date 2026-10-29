#!/usr/bin/env python3
"""Run one MLIR pass recipe on an exported Triton MLIR artifact."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ir_utils import find_tool, run_external_mlir_tool, write_text_artifact


RECIPES = {
    "ttir-cleanup": ["canonicalize", "cse", "symbol-dce"],
    "ttgir-cleanup": ["canonicalize", "cse", "symbol-dce"],
}
EXTERNAL_PIPELINES = {
    name: "builtin.module(" + ",".join(passes) + ")" for name, passes in RECIPES.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one MLIR pass recipe on a dumped Triton artifact. By default this uses "
            "Triton's Python pass manager; --backend external uses triton-opt/mlir-opt."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("artifacts/matmul/00_ttir.mlir"),
        help="Input MLIR file, for example artifacts/matmul/00_ttir.mlir.",
    )
    parser.add_argument(
        "--backend",
        choices=("internal", "external"),
        default="internal",
        help="Use Triton's Python pass manager or an external opt binary.",
    )
    parser.add_argument(
        "--tool",
        default="auto",
        help="External mode only: path/name of triton-opt or mlir-opt. Use 'auto' to search PATH.",
    )
    parser.add_argument(
        "--passes",
        default=None,
        help=(
            "Internal mode: comma-separated pass names such as canonicalize,cse. "
            "External mode: full MLIR pass pipeline."
        ),
    )
    parser.add_argument(
        "--recipe",
        choices=sorted(RECIPES),
        default="ttir-cleanup",
        help="Bundled teaching recipe used when --passes is omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write optimized MLIR. Default: <input stem>.<recipe>.mlir.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Where to write pass pipeline, stderr, and exit code.",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="External mode only: print the command that would be run.",
    )
    parser.add_argument(
        "--list-recipes",
        action="store_true",
        help="Print bundled recipes and exit.",
    )
    return parser.parse_args()


def parse_internal_passes(args: argparse.Namespace) -> list[str]:
    if args.passes:
        return [item.strip() for item in args.passes.split(",") if item.strip()]
    return RECIPES[args.recipe]


def run_internal(input_path: Path, pass_names: list[str], output_path: Path, log_path: Path) -> int:
    from triton._C import libtriton

    ctx = libtriton.ir.context()
    libtriton.ir.load_dialects(ctx)
    mod = libtriton.ir.parse_mlir_module(str(input_path), ctx)
    pm = libtriton.ir.pass_manager(ctx)

    adders = {
        "canonicalize": libtriton.passes.common.add_canonicalizer,
        "cse": libtriton.passes.common.add_cse,
        "symbol-dce": libtriton.passes.common.add_symbol_dce,
        "licm": libtriton.passes.common.add_licm,
        "sccp": libtriton.passes.common.add_sccp,
        "loop-aware-cse": libtriton.passes.ttir.add_loop_aware_cse,
        "triton-licm": libtriton.passes.ttir.add_triton_licm,
        "combine": libtriton.passes.ttir.add_combine,
    }

    unknown = [name for name in pass_names if name not in adders]
    if unknown:
        known = ", ".join(sorted(adders))
        raise SystemExit(f"Unknown internal pass(es): {', '.join(unknown)}. Known: {known}")

    for name in pass_names:
        adders[name](pm)

    pipeline = pm.get_pipeline_str()
    pm.run(mod, "builtin.module")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(mod.str(), encoding="utf-8")
    log_path.write_text(
        "\n".join(
            [
                "backend: internal",
                "passes: " + ",".join(pass_names),
                "pipeline: " + pipeline,
                "exit_code: 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return 0


def run_external(args: argparse.Namespace, input_path: Path, output_path: Path, log_path: Path) -> int:
    pipeline = args.passes or EXTERNAL_PIPELINES[args.recipe]
    tool = args.tool
    if tool == "auto":
        tool = find_tool(["triton-opt", "mlir-opt"]) or ""
    elif not Path(tool).exists():
        resolved = shutil.which(tool)
        tool = resolved or tool

    command = f"{tool or '<triton-opt|mlir-opt>'} {input_path} --pass-pipeline='{pipeline}'"
    if args.print_command:
        print(command)
        return 0

    if not tool:
        cmd_file = write_text_artifact(
            log_path.with_suffix(".cmd"),
            command
            + "\n\nNo triton-opt/mlir-opt binary was found on PATH. "
            "Install a Triton/LLVM developer build, then rerun this command.\n",
        )
        print(f"No triton-opt/mlir-opt found. Wrote command template to {cmd_file}")
        return 2

    return run_external_mlir_tool(tool, input_path, pipeline, output_path, log_path)


def main() -> int:
    args = parse_args()

    if args.list_recipes:
        for name, passes in RECIPES.items():
            print(f"{name}: {','.join(passes)}")
        return 0

    input_path = args.input.resolve()
    if not input_path.exists():
        raise SystemExit(f"Input MLIR file does not exist: {input_path}")

    output_path = args.output or input_path.with_name(f"{input_path.stem}.{args.recipe}.mlir")
    log_path = args.log or input_path.with_name(f"{input_path.stem}.{args.recipe}.log")

    if args.backend == "internal":
        pass_names = parse_internal_passes(args)
        rc = run_internal(input_path, pass_names, output_path, log_path)
        print("backend: internal")
        print(f"passes: {','.join(pass_names)}")
    else:
        rc = run_external(args, input_path, output_path, log_path)
        print("backend: external")

    print(f"output: {output_path}")
    print(f"log: {log_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
