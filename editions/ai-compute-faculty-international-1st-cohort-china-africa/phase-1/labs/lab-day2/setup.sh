#!/usr/bin/env bash
# setup.sh -- source this in every new terminal before running Day 2 scripts.
#
# Usage:
#   source setup.sh
#
# Day 2 shares the exact same environment as Day 1: CANN + torch_npu +
# triton-ascend 3.2.0. If you completed Day 1 and your shell can still
# import torch_npu and triton, you're already good.

# CANN env (same as Day 1)
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

echo "=== Day 2 environment ==="
python - <<'PY'
import sys
print(f"  Python      : {sys.version.split()[0]}")
try:
    import torch
    print(f"  PyTorch     : {torch.__version__}")
except Exception as e:
    print(f"  PyTorch     : MISSING ({e})")
try:
    import torch_npu
    print(f"  torch_npu   : {torch_npu.__version__}")
    print(f"  NPU avail   : {torch.npu.is_available()}  "
          f"(device count = {torch.npu.device_count()})")
except Exception as e:
    print(f"  torch_npu   : MISSING ({e})")
try:
    import triton
    print(f"  triton      : {triton.__version__}")
except Exception as e:
    print(f"  triton      : MISSING ({e})")
PY
echo "========================="
