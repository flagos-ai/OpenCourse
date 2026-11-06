# Harness Lab

# Download FlagOS components

```Bash
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

Ignore the incompatibility warnings

# Download model weights \(pre\-installed, can skip\)

FP16 full\-precision model

```Plain Text
modelscope download \
  qwen/Qwen2.5-14B-Instruct \
  --local-dir /flagos/lab/models/Qwen2.5-14B-Instruct
```

# Download and configure Hermes Agent \(pre\-installed, can skip\)

1. Download and install

```Plain Text
pip install hermes-agent
hermes postinstall
hermes --version
```

1. Choose "n" for all options until "Post\-install complete" appears\.

```Bash
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

1. Set the model

```Bash
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

Open config\.yaml and add supports\_tools: true

```YAML
model:
  provider: custom
  base_url: http://172.17.0.2:8010/v1
  api_key: not-needed
  default: qwen
  context_length: 65536
  max_tokens: 8192
  supports_tools: true
```

# **Start the service**

Start:

```Plain Text
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_PLUGINS=fl
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600

vllm serve /flagos/lab/models/Qwen2.5-14B-Instruct \
  --tensor-parallel-size 4 \
  --served-model-name qwen \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host 172.17.0.2 \
  --port 8010 \
  --enforce-eager \
  --max-model-len 65536
```

# Harness mini task

Start the hermes agent

```Plain Text
hermes
```

The prompt for the Hermes Agent:

```Plain Text
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

Note: The vllm 0\.24 hermes parser is incompatible with the Qwen2\.5 tool output format — it cannot parse tool\_calls and degrades to plain text, so tool calls do not work and the code is output as plain text\. You need to manually copy and paste the code into \.py files and converse with hermes to make all tests pass\.



# Open\-ended tasks

1. Design a transform LLM output hook to convert model output into a format the hermes parser can recognize\. Refer to the Event Hooks documentation:

> [https://hermes\-agent\.nousresearch\.com/docs/zh\-Hans/user\-guide/features/hooks](https://hermes-agent.nousresearch.com/docs/zh-Hans/user-guide/features/hooks)
> 
> 

2. None of the existing parsers work; you can design a new parser\.



# **Answer to Option 1: Configure the code\-writer hook \(optional fallback\)**

When Hermes can't turn Qwen2\.5's output into a `file` tool call, use a `transform_llm_output` hook to extract code blocks from the model response and write them to disk\.

## 1\. Place the script

Create the directory first \(it may not exist in a fresh container\):

```bash
mkdir -p /root/.hermes/agent-hooks
```

Put `code-writer.py` at:

```Plain Text
/root/.hermes/agent-hooks/code-writer.py
```

Make it executable \(copying from Windows strips the exec bit, so this is required\):

```bash
chmod +x /root/.hermes/agent-hooks/code-writer.py
```

## 2\. Update the config

Append to the end of `/root/.hermes/config.yaml` \(`hooks_auto_accept` must start at column 0, same level as `hooks` — do not indent it under `hooks`\):

```yaml
hooks:
  transform_llm_output:
    - command: "/root/.hermes/agent-hooks/code-writer.py"
      timeout: 15
hooks_auto_accept: true
```

## 3\. Smoke test

```bash
printf '%s' '{"cwd":"/tmp","extra":{"response_text":"x\n```python\n# todo.py\nprint(1)\n```\n"}}' | python3 /root/.hermes/agent-hooks/code-writer.py; cat /tmp/todo.py
```

You should see `[code-writer] wrote: ['todo.py']` and `print(1)`\.

## 4\. Authorize \+ verify \(must redo after switching containers/environments\)

The allowlist lives at `~/.hermes/shell-hooks-allowlist.json`; it is not migrated with the container, so you must re\-authorize in every new environment\.

**Important**: Both the authorize and verify commands must be run from `/root` \(i\.e\. `~`\); otherwise Hermes reads a different config/allowlist and `hooks doctor` keeps showing `✗ not allowlisted`:

```bash
cd /root
hermes --accept-hooks
hermes hooks doctor
```

`doctor` must show all three checks as ✓:

```Plain Text
✓ script exists and is executable
✓ allowlisted
✓ JSON smoke test
```

As long as you see `✗ not allowlisted`, the hook will not fire at runtime and no files will be produced\.

## 5\. Run the task

```bash
cd /root
hermes --accept-hooks
# send the todo prompt, wait for it to finish
ls -l /root/todo.py /root/test_todo.py
pytest /root/test_todo.py
```

## 6\. Troubleshooting when no files are produced

```bash
hermes logs --level INFO | grep code-writer
```

- If you see `[code-writer] wrote: [...]`: the hook fired\. No files on disk means the model didn't emit a recognizable ````python` block — inspect the raw model output\.

- If there's no output at all: the hook didn't fire\. Go back to step 4 and confirm `allowlisted`; verify the config indentation is correct and the script has the exec bit\.

## code\-writer\.py source

\[code\-writer\.py\]



> （注：部分内容由豆包工作 AI 生成）
