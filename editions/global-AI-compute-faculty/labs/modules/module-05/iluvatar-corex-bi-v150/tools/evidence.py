"""Small provenance helpers shared by the lab's result-producing steps."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint(root: Path) -> str:
    names = ('01_benchmark.py', 'loadgen/runner.py', 'loadgen/metrics.py',
             'kernels/gemm_triton.py', 'kernels/gemm_torch.py', 'tools/evidence.py')
    return hashlib.sha256(''.join(name + file_sha256(root / name) for name in names).encode()).hexdigest()


def sut_signature(config: dict) -> dict:
    """Only execution-relevant fields; labels and measured validation errors vary."""
    return {key: config.get(key) for key in
            ('backend', 'device', 'dtype', 'shape', 'tile', 'num_warps', 'num_stages')}
