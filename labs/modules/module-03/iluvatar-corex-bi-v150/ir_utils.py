"""Utilities for exporting Triton compiler artifacts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


STAGE_FILENAMES = {
    "ttir": "00_ttir.mlir",
    "ttgir": "01_ttgir.mlir",
    "llir": "02_llir.ll",
    "llvmir": "02_llvmir.ll",
    "ptx": "03_ptx.ptx",
    "cubin": "04_cubin.bin",
}


def prepare_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_triton_cache(dump_dir: Path, clear_cache: bool) -> Path:
    """Use an experiment-local cache so JIT recompilation is predictable."""

    cache_dir = dump_dir / ".triton-cache"
    if clear_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    return cache_dir


def export_compiled_artifacts(compiled: Any, out_dir: Path) -> dict[str, str]:
    """Write all available textual artifacts from a Triton CompiledKernel."""

    prepare_output_dir(out_dir)
    asm = getattr(compiled, "asm", None)
    if not asm:
        raise RuntimeError(
            "Triton did not expose compiled.asm. Try upgrading to Triton 3.x "
            "or enable MLIR_ENABLE_DUMP=1 / LLVM_IR_ENABLE_DUMP=1."
        )

    exported: dict[str, str] = {}
    for stage, artifact in asm.items():
        filename = STAGE_FILENAMES.get(stage, f"{stage}.txt")
        path = out_dir / filename
        if isinstance(artifact, bytes):
            path.write_bytes(artifact)
        else:
            path.write_text(str(artifact), encoding="utf-8")
        exported[stage] = str(path)
    return exported


def write_run_metadata(out_dir: Path, metadata: dict[str, Any]) -> Path:
    path = out_dir / "run.json"
    path.write_text(
        json.dumps(to_jsonable(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def write_text_artifact(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def find_tool(candidates: list[str]) -> str | None:
    for name in candidates:
        tool = shutil.which(name)
        if tool:
            return tool
    return None


def run_external_mlir_tool(
    tool: str,
    input_path: Path,
    pass_pipeline: str,
    output_path: Path,
    log_path: Path,
) -> int:
    """Run triton-opt/mlir-opt on one exported MLIR file."""

    cmd = [tool, str(input_path), f"--pass-pipeline={pass_pipeline}"]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(proc.stdout, encoding="utf-8")
    log_path.write_text(
        "\n".join(
            [
                "$ " + " ".join(cmd),
                f"exit_code: {proc.returncode}",
                "",
                "stderr:",
                proc.stderr,
            ]
        ),
        encoding="utf-8",
    )
    return proc.returncode


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))
    return repr(value)


def dump_hint() -> str:
    return (
        "For pass-by-pass traces, rerun with MLIR_ENABLE_DUMP=1 and/or "
        "LLVM_IR_ENABLE_DUMP=1. The lab already sets TRITON_CACHE_DIR inside "
        "the artifact directory so cache state is easy to inspect or remove."
    )
