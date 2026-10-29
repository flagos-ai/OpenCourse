"""Device helpers shared by GEMM SUTs."""

from __future__ import annotations

from typing import Any


def resolve_device(prefer: str | None = None) -> str:
    """Return best available torch device string."""
    import torch

    if prefer:
        if prefer == "cuda" and torch.cuda.is_available():
            return "cuda"
        if prefer == "cpu":
            return "cpu"
        if prefer.startswith("cuda") and torch.cuda.is_available():
            return prefer
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def synchronize(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def dtype_from_name(name: str) -> Any:
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def sizeof_dtype(name: str) -> int:
    mapping = {
        "float16": 2,
        "fp16": 2,
        "float32": 4,
        "fp32": 4,
        "bfloat16": 2,
        "bf16": 2,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def hardware_info() -> dict:
    import platform
    import torch

    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": None,
        "device_count": 0,
    }
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
    return info
