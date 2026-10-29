"""Ensure CoreX library env is present before importing torch.

LD_PRELOAD only takes effect if set before process start, so callers should
re-exec after applying these settings.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple


def _newest_corex_home() -> Optional[str]:
    if os.path.isdir("/usr/local/corex"):
        return "/usr/local/corex"
    matches = sorted(glob.glob("/usr/local/corex-*"))
    return matches[-1] if matches else None


def find_libcuda(corex_home: Optional[str]) -> Optional[str]:
    candidates: List[str] = []
    if corex_home:
        candidates.extend(
            [
                f"{corex_home}/lib64/libcuda.so.1",
                f"{corex_home}/lib64/libcuda.so",
            ]
        )
    candidates.append("/usr/local/iluvatar/lib64/libcuda.so.1")
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def prepare_corex_environ(env: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, str], List[str]]:
    """Return (new_env, change_notes). Does not mutate the process env."""
    out = dict(env if env is not None else os.environ)
    notes: List[str] = []

    home = out.get("COREX_HOME") or _newest_corex_home()
    if home and out.get("COREX_HOME") != home:
        out["COREX_HOME"] = home
        notes.append(f"COREX_HOME={home}")

    if home:
        lib64 = f"{home}/lib64"
        py_dist = f"{lib64}/python3/dist-packages"
        ld = out.get("LD_LIBRARY_PATH", "")
        if lib64 not in ld.split(":"):
            out["LD_LIBRARY_PATH"] = f"{lib64}:{ld}" if ld else lib64
            notes.append(f"prepend LD_LIBRARY_PATH {lib64}")
        if os.path.isdir(py_dist):
            pp = out.get("PYTHONPATH", "")
            if py_dist not in pp.split(":"):
                out["PYTHONPATH"] = f"{py_dist}:{pp}" if pp else py_dist
                notes.append(f"prepend PYTHONPATH {py_dist}")

    if os.path.isdir("/usr/local/iluvatar/lib64"):
        ld = out.get("LD_LIBRARY_PATH", "")
        if "/usr/local/iluvatar/lib64" not in ld.split(":"):
            out["LD_LIBRARY_PATH"] = f"/usr/local/iluvatar/lib64:{ld}" if ld else "/usr/local/iluvatar/lib64"
            notes.append("prepend /usr/local/iluvatar/lib64")

    libcuda = find_libcuda(out.get("COREX_HOME"))
    if libcuda:
        preload = out.get("LD_PRELOAD", "")
        parts = [p for p in preload.split(":") if p]
        if libcuda not in parts:
            out["LD_PRELOAD"] = libcuda if not parts else f"{libcuda}:{preload}"
            notes.append(f"LD_PRELOAD={libcuda}")

    return out, notes


def needs_reexec_for_corex() -> bool:
    """True if LD_PRELOAD/COREX_HOME are missing but can be filled."""
    if os.environ.get("SMOKE_COREX_REEXEC") == "1":
        return False
    _, notes = prepare_corex_environ()
    # Only re-exec when we would add LD_PRELOAD (must be before process start).
    return any(n.startswith("LD_PRELOAD=") or n.startswith("COREX_HOME=") for n in notes)
