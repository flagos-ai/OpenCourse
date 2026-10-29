"""Triton tiled GEMM SUT — used on GPU when Triton is available.

GEMM and validation are integrated from sunxt-0719/lab1 lab/common.py.
Native Triton autotuning follows lab/08_autotune.py; candidates come from
this repository's JSON configs. See docs/GEMM_SOURCE.md for provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .gemm_torch import TorchGEMM, _resolve_dtype

_TRITON_OK = False
try:
    import triton
    import triton.language as tl

    _TRITON_OK = True
except Exception:  # pragma: no cover - optional dependency
    triton = None  # type: ignore
    tl = None  # type: ignore

DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 2


def triton_available() -> bool:
    return _TRITON_OK and torch.cuda.is_available()


def _log(msg: str) -> None:
    print(msg, flush=True)


if _TRITON_OK:

    # Same calculation as lab1/lab/common.py::matmul_kernel (6004a08).
    @triton.jit
    def _matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, tl.cdiv(K, BLOCK_K)):
            remain = K - k_start * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < remain),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < remain) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_fixed(a, b, block_m=32, block_n=32, block_k=32,
                 num_warps=DEFAULT_NUM_WARPS, num_stages=DEFAULT_NUM_STAGES,
                 out=None):
    """Integrated Lab1 fixed launch with reusable output and tail masks."""
    M, K = a.shape
    K2, N = b.shape
    if K != K2 or not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("GEMM needs matching K and contiguous inputs")
    if a.device != b.device or a.dtype != b.dtype:
        raise ValueError("GEMM inputs must have the same device and dtype")
    c = out if out is not None else torch.empty((M, N), device=a.device, dtype=a.dtype)
    if c.shape != (M, N) or c.dtype != a.dtype or c.device != a.device or not c.is_contiguous():
        raise ValueError("GEMM output must be contiguous and match shape, dtype, and device")
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    return c


def check_close(actual, a, b, expected=None):
    """Lab1 fp32 reference and fp16 tolerance, always outside measured queries."""
    if expected is None:
        expected = (a.float() @ b.float()).to(actual.dtype)
    if not torch.isfinite(actual).all().item():
        raise AssertionError("Triton output contains NaN or Inf")
    torch.testing.assert_close(actual, expected, atol=0.2, rtol=1e-2)
    return (actual - expected).abs().max().item()


@dataclass
class TritonGEMM:
    """System Under Test wrapping a Triton tiled GEMM kernel."""

    M: int
    N: int
    K: int
    dtype: str = "float16"
    device: Optional[str] = None
    tile: Dict[str, int] = field(
        default_factory=lambda: {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}
    )
    name: str = "triton"
    autotune_configs: Optional[List[Dict[str, int]]] = None
    num_warps: int = DEFAULT_NUM_WARPS
    num_stages: int = DEFAULT_NUM_STAGES

    def __post_init__(self) -> None:
        if not triton_available():
            raise RuntimeError(
                "Triton GEMM requires CUDA + triton. "
                "Use TorchGEMM fallback on this machine."
            )
        from .gemm_torch import init_cuda_device

        self._device = (
            torch.device(self.device)
            if self.device
            else init_cuda_device(0)
        )
        self._dtype = _resolve_dtype(self.dtype)
        self.num_warps = int(self.tile.get("num_warps", self.num_warps))
        self.num_stages = int(self.tile.get("num_stages", self.num_stages))
        self.tile = {key: int(self.tile[key]) for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K")}
        _log(f"[sut] allocate tensors on {self._device} shape=({self.M},{self.N},{self.K})")
        self.A = torch.randn(self.M, self.K, device=self._device, dtype=self._dtype)
        _log("[sut] A ready")
        self.B = torch.randn(self.K, self.N, device=self._device, dtype=self._dtype)
        _log("[sut] B ready")
        self.C = torch.empty(self.M, self.N, device=self._device, dtype=self._dtype)
        _log("[sut] C ready; synchronize ...")
        torch.cuda.synchronize()
        _log("[sut] allocate done")
        self._best_tile = dict(self.tile)
        self._autotuned = False
        self._compiled_once = False
        self._autotuned_kernel = None
        self._validation = None
        self._candidate_validation = []

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def _launch(self, tile: Dict[str, int]) -> None:
        assert _TRITON_OK
        bm, bn, bk = int(tile["BLOCK_M"]), int(tile["BLOCK_N"]), int(tile["BLOCK_K"])
        grid = (triton.cdiv(self.M, bm), triton.cdiv(self.N, bn))
        if not self._compiled_once:
            _log(
                f"[sut] first Triton launch (JIT compile may take several minutes on CoreX); "
                f"tile={bm}x{bn}x{bk} grid={grid} — GPU util can stay ~0% while compiling"
            )
        if self._autotuned_kernel is None:
            matmul_fixed(
                self.A, self.B, bm, bn, bk,
                num_warps=int(tile.get("num_warps", self.num_warps)),
                num_stages=int(tile.get("num_stages", self.num_stages)),
                out=self.C,
            )
        else:
            auto_grid = lambda meta: (triton.cdiv(self.M, meta["BLOCK_M"]),
                                      triton.cdiv(self.N, meta["BLOCK_N"]))
            self._autotuned_kernel[auto_grid](
                self.A, self.B, self.C, self.M, self.N, self.K,
                self.A.stride(0), self.A.stride(1),
                self.B.stride(0), self.B.stride(1),
                self.C.stride(0), self.C.stride(1),
            )
        if not self._compiled_once:
            self.synchronize()
            self._compiled_once = True
            _log("[sut] first launch finished (compile + sync ok)")

    def run_once(self) -> None:
        self._launch(self._best_tile)

    def validate(self) -> Dict[str, Any]:
        """Validate the selected SUT before LoadGen starts timing."""
        self.run_once()
        self.synchronize()
        diff = check_close(self.C, self.A, self.B)
        self._validation = {"status": "PASS", "max_abs_err": diff,
                            "atol": 0.2, "rtol": 1e-2,
                            "reference": "fp32 matmul rounded to output dtype"}
        return dict(self._validation)

    def autotune(self, configs: Optional[List[Dict[str, int]]] = None) -> Dict[str, int]:
        """Use Lab1's native Triton autotune after checking every candidate."""
        candidates = configs or self.autotune_configs or [self.tile]
        self._autotuned_kernel = None
        self._autotuned = False
        self._candidate_validation = []
        expected = (self.A.float() @ self.B.float()).to(self.C.dtype)
        native_configs = []
        for cfg in candidates:
            self._launch(cfg)
            self.synchronize()
            diff = check_close(self.C, self.A, self.B, expected)
            tile = {key: int(cfg[key]) for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K")}
            warps = int(cfg.get("num_warps", self.num_warps))
            stages = int(cfg.get("num_stages", self.num_stages))
            native_configs.append(triton.Config(tile, num_warps=warps, num_stages=stages))
            self._candidate_validation.append({"tile": tile, "num_warps": warps,
                                              "num_stages": stages, "status": "PASS",
                                              "max_abs_err": diff})
        self._autotuned_kernel = triton.autotune(
            configs=native_configs, key=["M", "N", "K"]
        )(_matmul_kernel)
        self._launch(self._best_tile)
        self.synchronize()
        best = self._autotuned_kernel.best_config
        self._best_tile = dict(best.kwargs)
        self.tile = dict(self._best_tile)
        self.num_warps, self.num_stages = best.num_warps, best.num_stages
        self._autotuned = True
        self.validate()
        _log(f"[sut] native Triton autotune winner: {best}")
        return dict(self._best_tile)

    def config(self) -> Dict[str, Any]:
        return {
            "implementation": self.name,
            "backend": "triton",
            "source": "lab1-integrated@6004a08",
            "autotune_method": "triton.autotune" if self._autotuned else None,
            "validation": self._validation,
            "candidate_validation": list(self._candidate_validation),
            "device": str(self._device),
            "dtype": str(self._dtype).replace("torch.", ""),
            "shape": {"M": self.M, "N": self.N, "K": self.K},
            "tile": dict(self._best_tile),
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "autotuned": self._autotuned,
        }

    def flops(self) -> int:
        return 2 * self.M * self.N * self.K

    def bytes_moved(self) -> int:
        elem = torch.tensor([], dtype=self._dtype).element_size()
        return (self.M * self.K + self.K * self.N + self.M * self.N) * elem


def build_sut(
    kind: str,
    M: int,
    N: int,
    K: int,
    dtype: str,
    baseline_tile: Dict[str, int],
    autotune_configs: Optional[List[Dict[str, int]]] = None,
) -> Tuple[Any, str]:
    """Build local baseline/autotune GEMM or the PyTorch comparison SUT."""
    kind = kind.lower()
    if kind == "torch":
        _log("[sut] building torch.matmul SUT")
        return TorchGEMM(M=M, N=N, K=K, dtype=dtype, name="torch"), "torch"

    if not triton_available():
        return (
            TorchGEMM(M=M, N=N, K=K, dtype=dtype, name=f"torch_fallback_{kind}"),
            f"torch_fallback_{kind}",
        )

    if kind == "baseline":
        _log(f"[sut] building built-in Triton baseline tile={baseline_tile}")
        sut = TritonGEMM(
            M=M, N=N, K=K, dtype=dtype, tile=dict(baseline_tile), name="baseline"
        )
        return sut, "baseline"

    if kind == "autotune":
        _log("[sut] building integrated Lab1 GEMM + native Triton autotune")
        sut = TritonGEMM(
            M=M,
            N=N,
            K=K,
            dtype=dtype,
            tile=dict(baseline_tile),
            name="autotune",
            autotune_configs=autotune_configs,
        )
        sut.autotune(autotune_configs)
        return sut, "autotune"

    raise ValueError(f"unknown SUT kind: {kind}")
