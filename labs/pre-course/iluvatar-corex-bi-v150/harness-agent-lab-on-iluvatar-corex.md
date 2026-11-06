# Harness Agent Lab on Iluvatar CoreX BI-V150

**Course relationship:** pre-course lab for the *AI Large Language Model General Course*.
**Platform:** `iluvatar-corex-bi-v150`.

This lab serves Qwen2.5-14B-Instruct with vLLM, the FlagOS vLLM plugin, and FlagGems, enables tool calling, and drives it with the Hermes agent harness to complete a small autonomous coding task end to end.

**Note:** due to compatibility issues between the model and the Hermes parser (see [Harness mini task](#harness-mini-task)), this experiment requires manually creating the scripts to complete the entire harness experiment.

The course environment has the required packages preinstalled. Run the install section below only when preparing a new environment.

## Download FlagOS components

```bash
# 1. Build the vLLM v0.24.0 empty variant
git clone https://gitcode.com/vllm-project/vllm.git
cd ./vllm
git checkout v0.24.0
pip install setuptools_scm
VLLM_TARGET_DEVICE=empty pip install -v --no-build-isolation --no-deps .

# 2. Install vllm-plugin-FL (the FlagOS integration plugin)
cd ..
git clone https://gitcode.com/flagos-ai/vllm-plugin-FL -b main
cd vllm-plugin-FL
pip install --no-build-isolation --no-deps -e .

# 3. Install the FlagGems operator library v5.3.4
cd ..
git clone https://gitcode.com/flagos-ai/FlagGems
cd FlagGems
git checkout v5.3.4
pip install --no-build-isolation .
```

Ignore the incompatibility warnings.

## Download model weights

FP16 full-precision model:

```bash
modelscope download \
  qwen/Qwen2.5-14B-Instruct \
  --local-dir /models/Qwen2.5-14B-Instruct
```

## Download and configure the Hermes agent

1. Download and install:

```bash
pip install hermes-agent
hermes postinstall
hermes --version
```

2. Choose "n" for all options until "Post-install complete" appears:

```text
Node.js (required for browser tools and TUI) is not installed. Install now? [Y/n] n
Browser engine (Chromium, for web browsing tools) is not installed. Install now? [Y/n] n
ripgrep (fast file search) is not installed. Install now? [Y/n] n
ffmpeg (TTS voice messages) is not installed. Install now? [Y/n] n

✓ Post-install complete.
Hermes Agent v0.19.0 (2026.7.20)
Install directory: /usr/local/lib/python3.12/site-packages
Install method: pip
Python: 3.12.11
OpenAI SDK: 2.24.0
Up to date
```

3. Set the model:

```bash
hermes config set model.provider custom
hermes config set model.base_url http://172.17.0.2:8010/v1
hermes config set model.api_key not-needed
hermes config set model.default qwen
hermes config set model.context_length 65536
hermes config set model.max_tokens 8192
hermes config set agent.tool_use_enforcement true
hermes config get agent.tool_use_enforcement
hermes tools enable file
hermes tools enable terminal
```

4. Open `config.yaml` and add `supports_tools: true`:

```yaml
model:
  provider: custom
  base_url: http://172.17.0.2:8010/v1
  api_key: not-needed
  default: qwen
  context_length: 65536
  max_tokens: 8192
  supports_tools: true
```

## Start the service

```bash
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_PLUGINS=fl
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600

vllm serve /models/Qwen2.5-14B-Instruct \
  --tensor-parallel-size 4 \
  --served-model-name qwen \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host 172.17.0.2 \
  --port 8010 \
  --enforce-eager \
  --max-model-len 65536
```

## Harness mini task

Start the Hermes agent:

```bash
hermes
```

Give the Hermes agent this prompt:

```text
Please complete the following tasks:
Create todo.py that implements the following features:
add "task description" — add a todo item
list — list all todos (with index and status)
done <index> — mark a todo as done
remove <index> — delete a todo item
Persist the data to a todos.json file
Create test_todo.py that covers the following test cases:
After adding a todo, list shows it correctly
After marking a todo as done, its status becomes ✅
After deletion, list no longer contains the item
Operating on a non-existent index returns an error
Run pytest test_todo.py -v and make sure all tests pass
Finally report: how many files were created and how many tests passed
```

**Parser incompatibility:** the vLLM v0.24 `hermes` tool parser is incompatible with the Qwen2.5 tool output format. It cannot parse `tool_calls` and degrades them to plain text, so tool calls do not work and the agent outputs code as plain text. Copy each code block the agent prints into `todo.py` and `test_todo.py` yourself, and keep conversing with the agent until all tests pass.

## Completion criteria

Manually create `todo.py` and `test_todo.py` from the code the agent produces, run `pytest test_todo.py -v`, and iterate with the agent until every test passes. Record the model, endpoint, platform, the agent's final report, the code you pasted in, and any runtime error in your lab notes.

## Open-ended tasks

1. Design a transform LLM output hook that converts the model output into a format the `hermes` parser can recognize. See the Event Hooks documentation:
   <https://hermes-agent.nousresearch.com/docs/zh-Hans/user-guide/features/hooks>
2. None of the existing parsers work; you can design a new parser.
