# Image-to-Text Inference Lab on Iluvatar CoreX BI-V150

**Course relationship:** pre-course lab for the *AI Large Language Model General Course*.  
**Platform:** `iluvatar-corex-bi-v150`.

This is the maintained learner-facing manual. It runs Qwen3.6-35B-A3B with vLLM, the FlagOS vLLM plugin, and FlagGems on the Iluvatar CoreX BI-V150 course environment.

## Environment

- Model weights: `/models/Qwen3.6-35B-A3B/`
- Runtime: Ubuntu 22.04.5, Python 3.12, vLLM 0.24.0, FlagGems v5.3.4
- Accelerator: two BI-V150 cards / four logical devices

The course environment has the required packages preinstalled. Follow the dependency-installation section in [README.md](README.md) only when preparing a new environment.

## Start the service

In the first terminal:

```bash
export VLLM_PLUGINS=fl
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600

vllm serve /models/Qwen3.6-35B-A3B/ \
  -tp 4 \
  --served-model-name qwen \
  --port 8010 \
  --enforce-eager \
  --max-model-len 131072
```

The OpenAI-compatible endpoint is `http://localhost:8010/v1`.

## Online-image exercise

Create `client_image.py`:

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8010/v1")
response = client.chat.completions.create(
    model="qwen",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"
                },
            },
            {"type": "text", "text": "Describe the image and solve the illustrated geometry problem."},
        ],
    }],
    max_tokens=256,
    extra_body={"top_k": 20},
)
print(response.choices[0].message.content)
```

Run `python3 client_image.py` and record the response.

## Local-image exercise

Place `test.jpg` in the current directory and create `test_local_image.py`:

```python
import base64
from openai import OpenAI

with open("test.jpg", "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode("utf-8")

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8010/v1")
response = client.chat.completions.create(
    model="qwen",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Please identify and describe this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
        ],
    }],
    max_tokens=1024,
    extra_body={"top_k": 20},
)
print(response.choices[0].message.content)
```

Run `python3 test_local_image.py`.

## Completion criteria

The service remains healthy while both clients return a coherent response. Record the model path, endpoint, image source, response, and any issue. For connection errors, check the service terminal and port `8010`; for backend errors, verify `VLLM_PLUGINS=fl` and the installed runtime versions.
