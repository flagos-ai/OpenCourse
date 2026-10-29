#!/usr/bin/env bash
# Source on FlagOS / BI-V150 before lab scripts:
#   source ~/day2-lab5/env_corex.sh
#
# Some CoreX 4.4 images report torch.__version__ as plain "2.7.1"
# even when installed under /usr/local/corex-* — detect by install path.

# Do not use `set -e` here: this file is meant to be sourced.

if [[ -d /usr/local/corex ]]; then
  export COREX_HOME=/usr/local/corex
elif ls -d /usr/local/corex-* >/dev/null 2>&1; then
  export COREX_HOME="$(ls -d /usr/local/corex-* | sort -V | tail -n1)"
fi

if [[ -n "${COREX_HOME:-}" ]]; then
  export LD_LIBRARY_PATH="${COREX_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  if [[ -d "${COREX_HOME}/lib64/python3/dist-packages" ]]; then
    export PYTHONPATH="${COREX_HOME}/lib64/python3/dist-packages:${PYTHONPATH:-}"
  fi
  echo "[env_corex] COREX_HOME=${COREX_HOME}"

  for cand in \
      "${COREX_HOME}/lib64/libcuda.so.1" \
      "${COREX_HOME}/lib64/libcuda.so" \
      /usr/local/iluvatar/lib64/libcuda.so.1; do
    if [[ -f "${cand}" ]]; then
      export LD_PRELOAD="${cand}${LD_PRELOAD:+:$LD_PRELOAD}"
      echo "[env_corex] LD_PRELOAD=${cand}"
      break
    fi
  done
else
  echo "[env_corex] WARN: /usr/local/corex* not found (OK on local CPU)"
fi

if [[ -d /usr/local/iluvatar/lib64 ]]; then
  export LD_LIBRARY_PATH="/usr/local/iluvatar/lib64:${LD_LIBRARY_PATH:-}"
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

# Do NOT set CUDA_VISIBLE_DEVICES from /dev/iluvatarN (node id ≠ CUDA index).
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[env_corex] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (preset)"
else
  echo "[env_corex] CUDA_VISIBLE_DEVICES unset (recommended on single-card containers)"
fi

if ls /dev/iluvatar* >/dev/null 2>&1; then
  echo "[env_corex] device nodes: $(ls -1 /dev/iluvatar* 2>/dev/null | tr '\n' ' ')"
fi
