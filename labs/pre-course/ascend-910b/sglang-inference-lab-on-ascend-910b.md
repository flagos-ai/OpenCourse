# sglang Inference Lab on Ascend 910B

**Course relationship:** pre-course lab for the *AI Large Language Model General Course*.
**Platform:** `ascend-910b` (Huawei Ascend 910B).

This lab runs Qwen3.6-27B multimodal inference with sglang, the FlagOS sglang plugin, FlagGems, and FlagCX on Ascend 910B. It covers an offline smoke test, launching the sglang server with the correct operator blacklist, and driving it with a zero-dependency text/multimodal client.

The model weights are pre-downloaded at `/models/Qwen3.6-27B`.

## Inspect the FlagOS components

Enter the sglang workspace and list the installed FlagOS components:

```bash
cd /sgl-workspace
ls -la
```

Expected result:

```text
total 12
drwxr-xr-x  1 root root   115 Jul 23 02:58 .
drwxr-xr-x  1 root root    87 Oct 28 04:08 ..
drwxr-xr-x 16  501 staff 4096 Jul 22 13:18 FlagCX
drwxr-xr-x 21  501 staff 4096 Jul 10 03:42 FlagGems
drwxr-xr-x  1 root root    32 May  5 07:37 sglang
drwxr-xr-x  8 root root   245 Jul 23 02:58 sglang-plugin-FL
drwxr-xr-x  2 root root  4096 May 29 02:51 sgl-kernel-npu
drwxr-xr-x  2 root root    95 May 29 02:52 triton-ascend
```

## Offline smoke test

Enter the sglang-plugin-FL examples directory:

```bash
cd sglang-plugin-FL/examples
```

Disable the operators that are not supported on this backend and run a 4-TP offline inference pass:

```bash
SGLANG_FL_FLAGOS_BLACKLIST=index_copy_,gather,count_nonzero,index_put_,_index_put_impl_,cumsum,fill_scalar_,argmax,index,add_,conv1d,floor_divide,true_divide_ python3 qwen3_6_27b_offline_inference.py
```

The multimodal answers print first:

```text
[red_square.jpg] What color is shown in this image? Answer with one word.
    → ' Red'
  [cat.jpg] What animal is in this image? Answer with one word.
    → ' cat'
  [stop_sign.png] What is the color of this sign? Answer with one word.
    → ' red'
  [digit_seven.png] What digit is shown in this image? Answer with one digit.
    → '7'
```

The offline example may still raise a `kill_process_tree` / `resource_tracker` teardown warning after the answers are produced. That shutdown noise does not affect the inference results above.

## Launch the sglang server

Query the local IP address:

```bash
hostname -I
```

Run the launch command in `sglang-plugin-FL/examples`:

```bash
export SGLANG_FL_FLAGOS_BLACKLIST="index_copy_,gather,count_nonzero,index_put_,_index_put_impl_,cumsum,fill_scalar_,argmax,index,add_,conv1d,floor_divide,true_divide_"

# Before starting the server, make sure the variable was really exported
env | grep -i BLACKLIST

python3 -m sglang.launch_server \
  --model-path /models/Qwen3.6-27B \
  --host <localhost IP> --port 30000 \
  --tensor-parallel-size 4 --mem-fraction-static 0.7 \
  --disable-radix-cache --disable-cuda-graph \
  --trust-remote-code --reasoning-parser qwen3 --enable-multimodal
```

The server is ready when the log shows:

```text
[2026-10-28 09:48:38] INFO:     <localhost IP>:55878 - "POST /v1/chat/completions HTTP/1.1" 200 OK
[2026-10-28 09:48:38] The server is fired up and ready to roll!
```

Parameter explanation:

| Parameter | Meaning |
| --- | --- |
| `--model-path /models/Qwen3.6-27B` | Path to the model weights directory; required |
| `--host <localhost IP>` | Address the service listens on; inside a container prefer `0.0.0.0`, otherwise external access or the proxy may not connect |
| `--port 30000` | HTTP service port; defaults to 30000 |
| `--tensor-parallel-size 4` | TP=4, use 4 NPUs for tensor parallelism |
| `--mem-fraction-static 0.7` | Allocate about 70% of available memory to model weights + KV cache pool; default is 0.9, lower it on multi-device 27B setups to avoid OOM |
| `--disable-radix-cache` | Disable RadixAttention prefix caching; Huawei sglang 0.5.11 does not support prefix caching |
| `--disable-cuda-graph` | Disable CUDA Graph acceleration; commonly used when facing compatibility issues |
| `--trust-remote-code` | Allow executing remote code from the model repository; often required for non-official models |
| `--reasoning-parser qwen3` | Use the Qwen3 reasoning output parser to correctly parse markers such as `<think>` |
| `--enable-multimodal` | Enable multimodal support, allowing image input |

## Create the client

Open a new terminal and generate the zero-dependency client `sglang_client.py`:

```bash
cat << 'PYEOF' > sglang_client.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Zero-dependency sglang client: shared logic for text / multimodal (OpenAI-compatible /v1/chat/completions)
import argparse, base64, json, mimetypes, os, sys, urllib.request, urllib.error

DEFAULT_BASE = "http://localhost:30000"   # change to the proxy address when going through the platform proxy

def http_json(url, method="GET", payload=None, timeout=600):
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_models(base):
    return http_json(base.rstrip("/") + "/v1/models")

def auto_model(base):
    try:
        return get_models(base)["data"][0]["id"]
    except Exception:
        return "default"

def image_to_data_url(p):
    if p.startswith("http://") or p.startswith("https://"):
        return p
    if not os.path.exists(p):
        sys.exit("[Error] Image file does not exist: %s" % p)
    mime = mimetypes.guess_type(p)[0] or "image/jpeg"
    with open(p, "rb") as f:
        b = base64.b64encode(f.read()).decode("ascii")
    return "data:%s;base64,%s" % (mime, b)

def build_messages(text, image, system):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    if image:
        content = [{"type": "text", "text": text or "Please describe this image."},
                   {"type": "image_url", "image_url": {"url": image_to_data_url(image)}}]
    else:
        content = text
    msgs.append({"role": "user", "content": content})
    return msgs

def chat(base, model, messages, stream, temperature, max_tokens):
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens, "stream": bool(stream)}
    url = base.rstrip("/") + "/v1/chat/completions"
    if not stream:
        d = http_json(url, "POST", payload)
        print(d["choices"][0]["message"]["content"]); return
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                delta = json.loads(chunk)["choices"][0]["delta"].get("content")
                if delta:
                    sys.stdout.write(delta); sys.stdout.flush()
            except Exception:
                pass
    print()

def main():
    p = argparse.ArgumentParser(description="sglang text/multimodal client (zero-dependency)")
    p.add_argument("--base-url", default=DEFAULT_BASE)
    p.add_argument("--model", default=None)
    p.add_argument("--text", default="")
    p.add_argument("--image", default=None, help="local image path or http(s) URL; providing it switches to multimodal")
    p.add_argument("--system", default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--health", action="store_true")
    p.add_argument("--models", action="store_true")
    a = p.parse_args()
    if a.health:
        try:
            http_json(a.base_url.rstrip("/") + "/health")
            print("[OK] Service healthy: %s" % a.base_url)
        except Exception as e:
            print("[Failed] Health check failed: %s" % e); sys.exit(1)
        return
    if a.models:
        print(json.dumps(get_models(a.base_url), ensure_ascii=False, indent=2)); return
    if not a.text and not a.image:
        p.error("Please provide a question with --text (add --image for multimodal)")
    model = a.model or auto_model(a.base_url)
    chat(a.base_url, model, build_messages(a.text, a.image, a.system),
         a.stream, a.temperature, a.max_tokens)

if __name__ == "__main__":
    main()
PYEOF
```

Running the client with no arguments prints its usage:

```bash
python3 sglang_client.py
```

```text
usage: sglang_client.py [-h] [--base-url BASE_URL] [--model MODEL] [--text TEXT]
                        [--image IMAGE] [--system SYSTEM] [--temperature TEMPERATURE]
                        [--max-tokens MAX_TOKENS] [--stream] [--health] [--models]
sglang_client.py: error: Please provide a question with --text (add --image for multimodal)
```

## Text interaction

```bash
python3 sglang_client.py --base-url http://<localhost IP>:30000 --text "Explain in one sentence what KV Cache is."
```

## Image-to-text

Upload an image, then run the client against it:

```bash
python3 sglang_client.py --base-url http://<localhost IP>:30000 --image logo.png
```

## Completion criteria

The offline smoke test prints the four multimodal answers, the sglang server reaches "The server is fired up and ready to roll!", and the client returns a coherent response for both the text and image-to-text calls. Record the model path, endpoint, platform, the operator blacklist used, and any runtime error in your lab notes.
