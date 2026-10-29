"""Lightweight resource sampling (CPU / optional GPU via ixsmi or nvidia-smi)."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _run_cmd(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=5)
        return out.strip()
    except Exception:
        return None


def sample_gpu_once() -> Dict[str, Any]:
    """Prefer ixsmi (Iluvatar); fall back to nvidia-smi; else empty."""
    if shutil.which("ixsmi"):
        # Best-effort parse; formats vary by stack version.
        raw = _run_cmd(["ixsmi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"])
        if raw:
            parts = [p.strip() for p in raw.split(",")]
            sample: Dict[str, Any] = {"source": "ixsmi", "raw": raw}
            if len(parts) >= 1:
                try:
                    sample["gpu_util_pct"] = float(parts[0])
                except ValueError:
                    pass
            if len(parts) >= 2:
                try:
                    sample["mem_used_mb"] = float(parts[1])
                except ValueError:
                    pass
            if len(parts) >= 3:
                try:
                    sample["power_w"] = float(parts[2])
                except ValueError:
                    pass
            return sample
        return {"source": "ixsmi", "note": "ixsmi present but query failed"}

    if shutil.which("nvidia-smi"):
        raw = _run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ]
        )
        if raw:
            parts = [p.strip() for p in raw.split(",")]
            sample = {"source": "nvidia-smi", "raw": raw}
            try:
                sample["gpu_util_pct"] = float(parts[0])
                sample["mem_used_mb"] = float(parts[1])
                sample["power_w"] = float(parts[2])
            except (ValueError, IndexError):
                pass
            return sample

    return {"source": "none", "note": "no ixsmi/nvidia-smi available"}


@dataclass
class ResourceMonitor:
    interval_s: float = 0.2
    samples: List[Dict[str, Any]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None

    def start(self) -> None:
        try:
            import psutil  # noqa: F401
        except ImportError:
            self.samples.append({"source": "cpu", "note": "psutil not installed"})
            return

        self._stop.clear()

        def _loop() -> None:
            import psutil

            while not self._stop.is_set():
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory().percent
                gpu = sample_gpu_once()
                self.samples.append(
                    {
                        "t": time.time(),
                        "cpu_pct": cpu,
                        "mem_pct": mem,
                        "gpu": gpu,
                    }
                )
                time.sleep(self.interval_s)

        # Prime cpu_percent
        import psutil

        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.summarize()

    def summarize(self) -> Dict[str, Any]:
        if not self.samples:
            return {"sample_count": 0}
        cpu_vals = [s["cpu_pct"] for s in self.samples if "cpu_pct" in s]
        mem_vals = [s["mem_pct"] for s in self.samples if "mem_pct" in s]
        gpu_utils = [
            s["gpu"]["gpu_util_pct"]
            for s in self.samples
            if isinstance(s.get("gpu"), dict) and "gpu_util_pct" in s["gpu"]
        ]
        power_vals = [
            s["gpu"]["power_w"]
            for s in self.samples
            if isinstance(s.get("gpu"), dict) and "power_w" in s["gpu"]
        ]

        def _avg(xs: List[float]) -> Optional[float]:
            return float(sum(xs) / len(xs)) if xs else None

        return {
            "sample_count": len(self.samples),
            "cpu_pct_avg": _avg(cpu_vals),
            "mem_pct_avg": _avg(mem_vals),
            "gpu_util_pct_avg": _avg(gpu_utils),
            "power_w_avg": _avg(power_vals),
            "gpu_source": self.samples[-1].get("gpu", {}).get("source"),
        }
