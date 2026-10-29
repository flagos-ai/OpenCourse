# Image\-to\-Text Inference on Iluvatar CoreX — Step\-by\-Step Guide



This guide walks through the end\-to\-end steps to run image\-to\-text \(图生文\) inference using vLLM with the FlagOS plugin on an Iluvatar CoreX BI\-V150 platform\. The model used is Qwen3\.6\-35B\-A3B, with weights pre\-downloaded to `/models/Qwen3.6-35B-A3B/`\.

**Environment summary** — Iluvatar CoreX BI\-V150 \(2 cards / 4 cores, 32 GiB VRAM each\), Driver 4\.5\.0, IX\-ML 4\.4\.0, CUDA compatibility layer 10\.2, Ubuntu 22\.04\.5, Python 3\.12, vLLM 0\.24\.0, FlagGems v5\.3\.4\.

# Install FlagOS Components \(Pre\-intalled, skip this step\)

Clone and install vLLM \(empty\-device build\), the FlagOS vllm\-plugin\-FL and FlagGems\. Run the following commands:

```bash
git clone https://github.com/vllm-project/vllm.git
cd ./vllm
git checkout v0.24.0
pip install setuptools_scm
VLLM_TARGET_DEVICE=empty pip install -v --no-build-isolation --no-deps .

git clone https://github.com/flagos-ai/vllm-plugin-FL -b main
cd vllm-plugin-FL
pip install --no-build-isolation --no-deps -e .

git clone https://github.com/flagos-ai/FlagGems
cd FlagGems
git checkout v5.3.4
pip install --no-build-isolation .
```

# Start the Inference Service

Set the required environment variables and launch the vLLM serving engine with tensor parallelism across 4 cores\. The model weights are preinstalled and placed at  `/models/Qwen3.6-35B-A3B/`\.

```bash
export VLLM_PLUGINS=fl
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
vllm serve /models/Qwen3.6-35B-A3B/ -tp 4 --served-model-name qwen --port 8010 --enforce-eager --max-model-len 131072
```

The service listens on `http://localhost:8010` and exposes an OpenAI\-compatible API at `/v1`\.

# Test with an Online Image \(with simple pattern\)

The following script \(`client_image.py`\) sends a publicly accessible image URL together with a text question to the model via the OpenAI\-compatible API\. 

Start a new terminal, and run it with:

```bash
python3 client_image.py
```

Script content:

```python
from openai import OpenAI
# Configured by environment variables
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8010/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"
                }
            },
            {
                "type": "text",
                "text": "The centres of the four illustrated circles are in the corners of the square. The two big circles touch each other and also the two little circles. With which factor do you have to multiply the radii of the little circles to obtain the radius of the big circles?\nChoices:\n(A) $\\frac{2}{9}$\n(B) $\\sqrt{5}$\n(C) $0.8 \\cdot \\pi$\n(D) 2.5\n(E) $1+\\sqrt{2}$"
            }
        ]
    }
]
response = client.chat.completions.create(
    model="qwen",
    messages=messages,
    max_tokens=256,
    temperature=1.0,
    top_p=0.95,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
    },
    stream=False,
)

if response.choices and response.choices[0].message.content:
    print(response.choices[0].message.content)
```

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OGI1Y2Y1NmJiNzhmNDgxYzJjMjQ2YmEyMDlmYjk5NmRfNzgxZTliNDEzYWU3NGY0NmJhOTgyNzYwZjBlODE5NGFfSUQ6NzY4NTYwNDk0OTkxNjQzNzY5MF8xNzkxNzk1NjEzOjE3OTE4ODIwMTNfVjM)

A successful run prints the model's reasoning and answer to the geometry question in the terminal\.

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MWEzODJhOWU5YTQwYmM5OTI0ZjE3NGEwNjYxY2JhZmZfOTE3MTdiNTU2YTRhZWJkN2FhZTBiYjcyN2Y3MGUxN2RfSUQ6NzY4NTU3MzIwMjQ2MjU5MTk3Ml8xNzkxNzk1NjEzOjE3OTE4ODIwMTNfVjM)

# Test with a Local Image \(with complicated pattern\)

To test image recognition with a file stored locally, upload an image named `test.jpg` to the `/root/` directory\. The script \(`test_local_image.py`\) encodes the image as base64 and sends it inline in the request\.

Run it with:

```bash
python3 test_local_image.py
```

Script content:

```python
import base64
import os
from openai import OpenAI

# 1. Configure the client
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8010/v1"
)

# 2. Set the local image name to test (must match the file name placed under /root/)
local_image_path = "test.jpg"

# 3. Encode the local image as base64
try:
    with open(local_image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
except FileNotFoundError:
    print(f"Error: Image named '{local_image_path}' not found in the current directory. Please confirm the image has been placed under /root/.")
    exit()

# 4. Send the request
try:
    response = client.chat.completions.create(
        model="qwen",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please identify and describe the content of this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        max_tokens=1024,
        # Put extended parameters such as top_k into extra_body to avoid errors
        extra_body={"top_k": 20}
    )
    print("\n===== Model Response =====")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"\nAn error occurred: {e}")
```

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YzMyODdkNWJhYzI4OTNmOTJmMDliNmVlZTU1YTVjNmFfMmY5MjA2OTIzZmJjZTI4NGQ4OGU3MTQzZDE3ZmYzNTRfSUQ6NzY4NTYwNTgxNzQwMzc4ODQ2OV8xNzkxNzk1NjEzOjE3OTE4ODIwMTNfVjM)

A successful run prints the model's description of the local image content\.

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZWQ4ODI1N2YwMDFkNWRkNTk4ZTQ1ZTQ2MGEzYzEzNWJfOGE2MzAxNGM0YzhiZjhhMTk5ZDMwMDRmYmYxNGYzNDRfSUQ6NzY4NTU3MzMxNzA4OTAwNDUxOF8xNzkxNzk1NjEzOjE3OTE4ODIwMTNfVjM)



> （注：部分内容由豆包工作 AI 生成）
