# Image-to-Text Inference Lab on Iluvatar CoreX BI-V150

**Course relationship:** pre-course lab for the *AI Large Language Model General Course*.

**Platform:** `iluvatar-corex-bi-v150` (Iluvatar CoreX BI-V150).
**Maintenance:** this is the maintained mainline copy; editions are frozen snapshots.

This lab runs image-to-text inference with vLLM, the FlagOS vLLM plugin, and FlagGems. It uses the Qwen3.6-35B-A3B model preinstalled at `/models/Qwen3.6-35B-A3B/`.

For the stable learner-facing manual, use [image-to-text-inference-lab-on-iluvatar-corex.md](image-to-text-inference-lab-on-iluvatar-corex.md).

## Environment

- Iluvatar CoreX BI-V150: 2 cards / 4 logical cores, 32 GiB VRAM each
- Ubuntu 22.04.5, Python 3.12, vLLM 0.24.0, FlagGems v5.3.4
- Driver 4.5.0, IX-ML 4.4.0, CUDA compatibility layer 10.2

## Install FlagOS components

The course environment normally includes these dependencies already. For a new environment, install compatible versions:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm && git checkout v0.24.0
pip install setuptools_scm
VLLM_TARGET_DEVICE=empty pip install -v --no-build-isolation --no-deps .

git clone https://github.com/flagos-ai/vllm-plugin-FL -b main
cd vllm-plugin-FL && pip install --no-build-isolation --no-deps -e .

git clone https://github.com/flagos-ai/FlagGems
cd FlagGems && git checkout v5.3.4
pip install --no-build-isolation .
```

## Start the inference service

```bash
export VLLM_PLUGINS=fl
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
vllm serve /models/Qwen3.6-35B-A3B/ -tp 4 --served-model-name qwen --port 8010 --enforce-eager --max-model-len 131072
```

The OpenAI-compatible API is available at `http://localhost:8010/v1`.

## Exercise 1: online image

Create `client_image.py` with an OpenAI client pointing to `http://localhost:8010/v1`. Send an `image_url` content item and a text question, then run:

```bash
python3 client_image.py
```

## Exercise 2: local image

Place `test.jpg` in the working directory. Base64-encode it in `test_local_image.py`, send it as a `data:image/jpeg;base64,...` image URL, and run:

```bash
python3 test_local_image.py
```

For both exercises, confirm that the response contains a coherent description or answer and record the model, endpoint, platform, and any runtime error in your lab notes.
