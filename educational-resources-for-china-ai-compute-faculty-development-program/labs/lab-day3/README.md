# CANN-Triton Lab Day 3: LLM Deployment

Hands-on lab for the AI Compute Faculty Development Program.

Day 3 focuses on deploying and serving large language models (LLMs) locally. Building on Days 1-2, which covered NPU programming and performance optimization, today you will learn to deploy an LLM inference service and interact with it.

---

## Quick Start

### Prerequisites

- A running vLLM or compatible inference service at `http://localhost:30000/v1`
- Python 3.11+ with `openai` package installed

```bash
pip install openai
```

### Basic Usage

Simple single-turn chat:

```bash
python client.py
```

Continuous multi-turn conversation:

```bash
python test.py
```

---

## Files

| File | Description |
|------|-------------|
| `Practical_Handbook.pdf` | Detailed lab handbook with step-by-step instructions |
| `client.py` | Simple OpenAI-compatible client for single-turn chat |
| `test.py` | Multi-turn continuous chat client with context management |

---

## What You'll Learn

### 1. OpenAI-Compatible API

The local inference service exposes an OpenAI-compatible REST API. This means:

- You can use the official `openai` Python SDK
- Simple `base_url` change points to your local service
- Code written for OpenAI works with local models

### 2. Single-Turn vs Multi-Turn Chat

- `client.py` sends one message and receives one response
- `test.py` maintains conversation history for multi-turn dialogue

### 3. Context Management

The multi-turn client (`test.py`) includes:

- **Token estimation** - Rough count of CJK and English tokens
- **Context limit handling** - Automatic trimming when approaching limits
- **Interactive commands** - `/quit`, `/clear`, `/tokens`

---

## Client Usage

### test.py — Interactive Commands

```
You: Hello, who are you?
Qwen: I am a helpful AI assistant...
[~120/4096 tokens]

You: /tokens
[Used ~120 / 4096 tokens]

You: /clear
[History cleared]

You: /quit
Goodbye!
```

| Command | Description |
|---------|-------------|
| `/quit` | Exit the chat |
| `/clear` | Clear conversation history |
| `/tokens` | Show current token usage |

### Command-line Options

```bash
python test.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--base-url` | `http://localhost:30000/v1` | vLLM service URL |
| `--model` | `qwen` | Model name |
| `--max-tokens` | `512` | Max tokens per response |
| `--max-context` | `4096` | Total context window limit |
| `--temperature` | `0.7` | Generation temperature |
| `--system` | `You are a helpful assistant.` | System prompt |

---

## Environment

Same as Days 1-2:

| Component | Version |
|-----------|---------|
| Hardware | Ascend 910C (1 card, 2 dies) |
| CPU | aarch64 (Kunpeng / Phytium) |
| CANN | 8.3.RC2 |
| PyTorch | 2.8.0 (CPU build) |
| torch_npu | 2.8.0 |
| Triton-Ascend | 3.2.0 |
| Python | 3.11 |

---

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Connection refused` | Service not running | Start vLLM service on port 30000 |
| Empty response | Model not loaded | Wait for model initialization |
| `Context limit exceeded` | Long conversation | Use `/clear` or wait for auto-trim |
| Slow first response | Model loading | Subsequent responses are faster |

---

## Resources

- Day 1 lab: Ascend NPU & Triton basics
- Day 2 lab: Performance tuning and autotuning
- vLLM documentation: https://docs.vllm.ai/
- OpenAI API reference: https://platform.openai.com/docs/api-reference

---

## License

CC BY-NC 4.0 — Creative Commons Attribution-NonCommercial 4.0 International.
