#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import sys
import torch
import triton

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Triton:", triton.__version__)
print("Accelerator available:", torch.cuda.is_available())
print("Visible devices:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("BI-V150/CoreX device is not visible to PyTorch")
torch.cuda.set_device(0)
print("Selected device: cuda:0")
print("Device name:", torch.cuda.get_device_name(0))
print("Do not pip-install/upgrade torch or triton in this environment.")
PY
