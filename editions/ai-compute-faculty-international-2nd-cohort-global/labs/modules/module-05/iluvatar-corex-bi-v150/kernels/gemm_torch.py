"""PyTorch matmul SUT — baseline comparison and CPU/CUDA fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[key]


def pick_device(prefer: Optional[str] = None) -> torch.device:
    """Prefer explicit cuda:0 (matches lab1 DEVICE)."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def init_cuda_device(index: int = 0) -> torch.device:
    """Initialize CUDA context with visible progress (helps debug CoreX stalls)."""
    print(f"[cuda] available={torch.cuda.is_available()} count={torch.cuda.device_count()}", flush=True)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    print(f"[cuda] set_device({index}) ...", flush=True)
    torch.cuda.set_device(index)
    name = torch.cuda.get_device_name(index)
    print(f"[cuda] device_name={name}", flush=True)
    # Tiny sync to force context creation before large allocations.
    print("[cuda] probing with 1x1 tensor ...", flush=True)
    probe = torch.zeros(1, device=f"cuda:{index}", dtype=torch.float16)
    torch.cuda.synchronize()
    del probe
    print("[cuda] probe ok", flush=True)
    return torch.device(f"cuda:{index}")


@dataclass
class TorchGEMM:
    """System Under Test wrapping torch.matmul."""

    M: int
    N: int
    K: int
    dtype: str = "float16"
    device: Optional[str] = None
    name: str = "torch"

    def __post_init__(self) -> None:
        if self.device:
            self._device = torch.device(self.device)
            if self._device.type == "cuda":
                torch.cuda.set_device(self._device.index or 0)
                print(f"[sut:torch] using explicit device={self._device}", flush=True)
        else:
            self._device = init_cuda_device(0) if torch.cuda.is_available() else torch.device("cpu")

        self._dtype = _resolve_dtype(self.dtype)
        # CPU float16 matmul is often slow/unsupported paths; promote for correctness.
        if self._device.type == "cpu" and self._dtype == torch.float16:
            self._dtype = torch.float32

        print(
            f"[sut:torch] allocate A,B,C on {self._device} dtype={self._dtype} "
            f"shape=({self.M},{self.N},{self.K})",
            flush=True,
        )
        self.A = torch.randn(self.M, self.K, device=self._device, dtype=self._dtype)
        print("[sut:torch] A ready", flush=True)
        self.B = torch.randn(self.K, self.N, device=self._device, dtype=self._dtype)
        print("[sut:torch] B ready", flush=True)
        self.C = torch.empty(self.M, self.N, device=self._device, dtype=self._dtype)
        print("[sut:torch] C ready; synchronize ...", flush=True)
        self.synchronize()
        print("[sut:torch] allocate done", flush=True)

    def synchronize(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize()

    def run_once(self) -> None:
        # Preallocated out= path (same idea as Lab1 vendor comparison).
        torch.matmul(self.A, self.B, out=self.C)

    def config(self) -> Dict[str, Any]:
        return {
            "implementation": self.name,
            "backend": "torch.matmul",
            "device": str(self._device),
            "dtype": str(self._dtype).replace("torch.", ""),
            "shape": {"M": self.M, "N": self.N, "K": self.K},
        }

    def flops(self) -> int:
        return 2 * self.M * self.N * self.K

    def bytes_moved(self) -> int:
        elem = torch.tensor([], dtype=self._dtype).element_size()
        return (self.M * self.K + self.K * self.N + self.M * self.N) * elem
