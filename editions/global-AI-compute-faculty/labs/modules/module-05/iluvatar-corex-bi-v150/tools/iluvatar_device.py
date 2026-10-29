"""Iluvatar device-node helpers for FlagOS diagnostics.

Important: ``/dev/iluvatarN`` is a kernel device node name (often host GPU id).
Inside a container the CUDA runtime usually still enumerates the only mounted
card as logical index 0. Setting ``CUDA_VISIBLE_DEVICES=N`` from the node name
often hides the GPU entirely (``torch.cuda.is_available() == False``).
"""

from __future__ import annotations

import glob
import os
import re
from typing import List, Optional


def list_iluvatar_indices() -> List[int]:
    indices: List[int] = []
    for path in sorted(glob.glob("/dev/iluvatar[0-9]*")):
        m = re.search(r"iluvatar(\d+)$", path)
        if m:
            indices.append(int(m.group(1)))
    return indices


def describe_device_nodes() -> str:
    indices = list_iluvatar_indices()
    if not indices:
        return "no /dev/iluvatar* nodes"
    nodes = ", ".join(f"/dev/iluvatar{i}" for i in indices)
    return (
        f"{nodes} (node name ≠ CUDA index; prefer unset CUDA_VISIBLE_DEVICES "
        f"or 0, not {','.join(str(i) for i in indices)})"
    )


def ensure_cuda_visible_devices(force: bool = False) -> Optional[str]:
    """Compatibility stub: do NOT auto-set CVD from /dev/iluvatar*.

    Returns the current ``CUDA_VISIBLE_DEVICES`` value (may be None).
    """
    del force  # unused; kept for call-site compatibility
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    return existing if existing not in (None, "") else None


def maybe_clear_bad_cvd_hint() -> Optional[str]:
    """If CVD looks copied from iluvatar node numbers, return a warning string."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return None
    nodes = set(list_iluvatar_indices())
    if not nodes:
        return None
    try:
        cvd_ids = {int(x) for x in cvd.split(",") if x.strip() != ""}
    except ValueError:
        return None
    # Only iluvatar3 mounted but CVD=3 → almost certainly wrong in containers.
    if cvd_ids and cvd_ids == nodes and 0 not in nodes:
        return (
            f"CUDA_VISIBLE_DEVICES={cvd} matches /dev/iluvatar* node ids, but "
            "containers usually expose that card as CUDA index 0. "
            "Try: unset CUDA_VISIBLE_DEVICES"
        )
    return None
