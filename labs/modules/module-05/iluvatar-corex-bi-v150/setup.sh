#!/usr/bin/env bash
# Module-5 environment check (lab1 style).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Optional FlagOS / CoreX paths (no-op when /usr/local/corex* is absent).
if [[ -f "${ROOT}/env_corex.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/env_corex.sh" || true
fi

python3 - <<'PY'
import torch

try:
    import triton
    triton_ver = triton.__version__
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Triton import failed: {exc}") from exc

print("PyTorch:", torch.__version__)
print("Triton:", triton_ver)
print("torch file:", torch.__file__)
print("Accelerator available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        "BI-V150/CoreX device is not visible to PyTorch.\n"
        "For local CPU-only smoke, run: python3 00_check_env.py --allow-cpu"
    )
torch.cuda.set_device(0)
print("Selected device: cuda:0")
print("Device name:", torch.cuda.get_device_name(0))
print("Do not pip-install/upgrade torch or triton in this environment.")
PY

echo
echo "[setup] OK. Next:"
echo "  python3 00_check_env.py"
echo "  # On FlagOS, also: python3 00_smoke_cuda.py"
echo "  # GEMM is integrated locally; no external Lab1 checkout is required."
