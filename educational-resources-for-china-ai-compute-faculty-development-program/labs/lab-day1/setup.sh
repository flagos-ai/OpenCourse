#!/bin/bash
# Source this file at the start of every new terminal session.
# Usage:  source setup.sh

source /usr/local/Ascend/ascend-toolkit/set_env.sh

echo "======================================================"
echo "  CANN-Triton Lab Day 1 - Environment"
echo "======================================================"

CANN_VER=$(grep "toolkit_running" /usr/local/Ascend/ascend-toolkit/latest/version.cfg | head -1 | sed 's/.*\[\(.*\)\].*/\1/')
echo "CANN toolkit:    ${CANN_VER}"

python - <<'PY'
import torch, torch_npu, triton
print(f"PyTorch:         {torch.__version__}")
print(f"torch_npu:       {torch_npu.__version__}")
print(f"Triton-Ascend:   {triton.__version__}")
print(f"NPU available:   {torch.npu.is_available()}")
print(f"NPU device count: {torch.npu.device_count()}")
PY

echo "======================================================"
echo "  Ready. If you see all versions above and 'NPU"
echo "  available: True', your environment is OK."
echo "======================================================"
