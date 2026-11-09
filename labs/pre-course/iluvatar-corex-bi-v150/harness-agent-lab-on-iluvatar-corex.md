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

Note: ignore the incompatibility warnings

# Download model weights (pre-installed, can skip)

FP16 full-precision model

```Plain Text
modelscope download \
  qwen/Qwen2.5-14B-Instruct \
  --local-dir /models/Qwen2.5-14B-Instruct
```

# Download and configure Hermes Agent (pre-installed, can skip)

1. Download and install

```Plain Text
pip install hermes-agent
hermes postinstall
hermes --version
```

2. Choose "n" for all of the following options until "Post-install complete." appears.

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

3. Configure the config.yaml file

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

Open config.yaml and add supports_tools: true

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

# Harness mini task

Start the hermes agent

```Plain Text
hermes
```

The prompt for the Hermes Agent:

```Plain Text
请完成以下最小测试任务：

1. 写出 todo.py 的完整 Python 代码，要求：
   - 支持 add "任务描述"：添加一条待办；
   - 支持 list：列出所有待办，带编号和状态；
   - 支持 done 编号：标记为已完成；
   - 支持 remove 编号：删除一条待办；
   - 使用 argparse 或 sys.argv 解析命令行参数；
   

2. 写出test_todo.py 的完整 Python 测试代码，要求：
   - 测试添加待办后 list 能正确显示；
   - 测试标记完成后状态变为已完成；
   - 测试删除后 list 不再包含该条目；
   - 测试对不存在的编号操作时返回错误；
   - 使用 pytest 风格编写。
 

3. 最后只按以下格式汇报：
   - 创建了几个文件：
   - 预期 pytest 应该通过几项

```

# Problem:

Note: the vllm 0.24 hermes parser is incompatible with the Qwen2.5 tool output format. It cannot parse tool_call and degrades to plain text, so tool calls do not work and the code is output as plain text. You need to manually copy and paste it into .py files and converse with hermes to ultimately pass all tests.

Below are two approaches to complete automated Harness.

# Approaches (optional):

1. **Approach 1**: Design a transform LLM output hook to convert model output into a format the hermes parser can recognize. Refer to the Event Hooks documentation:

> [https://hermes-agent.nousresearch.com/docs/zh-Hans/user-guide/features/hooks](https://hermes-agent.nousresearch.com/docs/zh-Hans/user-guide/features/hooks)
> 
> 

2. **Approach 2**: None of the existing parsers work; you can design a new parser. Approach 2 also covers how to install the Holographic memory store + llm-wiki.

## **Approach 1: Configure the code-writer Hermes Hook**

Purpose: when Hermes cannot turn Qwen2.5 output into a `file` tool call, use a `transform_llm_output` hook to extract code blocks from the model response and write them to disk.

### 1. Place the script

Create the directory first (it may not exist in a fresh container):

```bash
mkdir -p /root/.hermes/agent-hooks
```

Put `code-writer.py` at:

```Plain Text
/root/.hermes/agent-hooks/code-writer.py
```

Add the executable bit (copying from Windows strips the exec bit, so this is required):

```bash
chmod +x /root/.hermes/agent-hooks/code-writer.py
```

### 2. Update the config

Append to the end of `/root/.hermes/config.yaml` (`hooks_auto_accept` must start at column 0, at the same level as `hooks` — do not indent it under `hooks`):

```yaml
hooks:
  transform_llm_output:
    - command: "/root/.hermes/agent-hooks/code-writer.py"
      timeout: 15
hooks_auto_accept: true
```

### 3. Self-test script

```bash
printf '%s' '{"cwd":"/tmp","extra":{"response_text":"x\n```python\n# todo.py\nprint(1)\n```\n"}}' | python3 /root/.hermes/agent-hooks/code-writer.py; cat /tmp/todo.py
```

It should output `[code-writer] wrote: ['todo.py']` and `print(1)`.

### 4. Authorize + verify (must be redone after switching containers/environments)

The allowlist file is at `~/.hermes/shell-hooks-allowlist.json`; it is not migrated with the container, so you must re-authorize in a new environment.

**Important**: both the authorize and verify commands must be run under `/root` (i.e. `~`); otherwise Hermes reads a config/allowlist that is not this one, and `hooks doctor` will keep showing `✗ not allowlisted`:

```bash
cd /root
hermes --accept-hooks
hermes hooks doctor
```

`doctor` must show all three items as ✓:

```Plain Text
✓ script exists and is executable
✓ allowlisted
✓ JSON smoke test
```

As long as it shows `✗ not allowlisted`, the hook will not fire at runtime and no files will be produced.

### 5. Run the task

```bash
cd /root
hermes --accept-hooks
# send the todo prompt, run it to completion and exit
ls -l /root/todo.py /root/test_todo.py
pytest /root/test_todo.py
```

### 6. Troubleshooting when no files are produced

```bash
hermes logs --level INFO | grep code-writer
```

- With `[code-writer] wrote: [...]`: the hook fired. If nothing was written to disk, the model did not emit a recognizable ````python` code block — inspect the raw model output.

- With no output at all: the hook did not fire. Return to step 4 and confirm `allowlisted`; confirm the config indentation is correct and the script has the exec bit.

### code-writer.py source

[code-writer.py]

## Approach 2: Design a new parser to replace the hermes parser, including installing the Holographic memory store

[hermes-vllm-toolcall-fix-guide](https://jwolpxeehx.feishu.cn/wiki/E6Abw8SdPieBYbkrfH3c2QRsncd)

---

# Hermes Holographic Memory Store — Installation and Structure Guide

## Why install Holographic

Hard limits: MEMORY.md ≤ 2200 characters, USER.md ≤ 1375 characters. Each Chinese character counts as 3 bytes; count with python len(content), not os.path.getsize. In practice about 700 / 450 Chinese characters.

- Only write third-person factual statements: "The user prefers X" / "This machine has no libreoffice", not "always does X". Imperative wording will be treated as a command to execute in future sessions and will override the user's current request.

- Trade-off rule: only cross-task general content goes into L1; specific people, projects, decisions, and channel details all go into L2. In L1, any occurrence of "see skill X" / "see so-and-so for details" — keeping a pointer is fine, but you must also confirm that the skill really will be migrated over.

- Do not pad to fill the budget; leave it empty.

## Installation / activation command

Enabling the holographic memory store takes just one command:

```bash
hermes memory setup holographic
```

On success it outputs `Activation saved to config.yaml`, and the provider is written to `~/.hermes/config.yaml` and activated.

### Companion commands

```bash
# View the current memory provider status (confirm whether it is active)
hermes memory status

# Without arguments, enter the interactive selector and manually pick holographic from the list
hermes memory setup

# Disable the external provider, returning to the built-in MEMORY.md only
hermes memory off

# Clear the built-in memory (MEMORY.md / USER.md); does not affect the holographic store
hermes memory reset
```

### Key points

- **No pip install of anything is needed**. The holographic plugin is installed together with hermes at
`/usr/local/lib/python3.12/site-packages/plugins/memory/holographic/`,
and depends only on numpy. `setup` only performs activation; it does not download.

- **The database is created automatically after setup, on the first memory write**, at the fixed path
`~/.hermes/memory_store.db`; there is no need to create it manually.

- holographic hooks into `on_session_end` — it only writes to disk when the session ends and there really is something to record.
So not seeing the .db with `ls` right after setup is **normal behavior**, not an error.

---

## Memory store structure

The memory store = a single local SQLite file `~/.hermes/memory_store.db`, with a three-layer architecture:
fact storage + entity graph + HRR holographic vector retrieval.

### 3.1 Core table: `facts` (the main memory table, one row per memory)

|Field|Meaning|
|---|---|
|`fact_id`|Primary key, auto-increment|
|`content`|Memory body (**UNIQUE**, content is deduplicated, the same sentence is stored only once)|
|`category`|Category, default `general`|
|`tags`|Tag string|
|`trust_score`|Trust score, default 0.5|
|`retrieval_count`|Number of times retrieved|
|`helpful_count`|Number of times judged helpful|
|`created_at` / `updated_at`|Timestamps|
|`hrr_vector`|BLOB, HRR holographic vector (1024-dim float64)|

> `trust_score` / `retrieval_count` / `helpful_count` make the store **self-reinforce with use**:
> memories that are hit more and found helpful more have higher scores and are prioritized during retrieval.
> 
> 

### 3.2 Layered architecture

**Layer 1 — Fact storage + full-text search**

- `facts` stores the body text

- `facts_fts` (FTS5 virtual table) + the companion tables `facts_fts_config/data/docsize/idx` build a full-text index over `content`/`tags`

- 3 triggers `facts_ai` / `facts_au` / `facts_ad` (insert/update/delete) keep the full-text index in sync automatically

- Purpose: keyword search

**Layer 2 — Entity knowledge graph**

- `entities` — entities extracted from memories (`name`, `entity_type`, `aliases`, `created_at`)

- `fact_entities` — a many-to-many association table recording "which memory mentions which entity"

- Purpose: link related memories into a graph by entity

**Layer 3 — HRR holographic vector retrieval** (the origin of the provider's name)

- `facts.hrr_vector` — the phase vector of each memory, deterministically generated via SHA-256, reproducible across machines / processes / language versions

- `memory_banks` — "memory-bank vectors" aggregated by `category`, bundling one category of memories into a composite vector

    - Fields: `bank_id` / `bank_name` (UNIQUE) / `vector` (BLOB) / `dim` / `fact_count` / `updated_at`

- Vector symbolic operations: `bind` (circular convolution = phase addition, associate concepts) / `unbind` (circular correlation = phase subtraction, retrieve) / `bundle` (superposition = circular mean, merge)

- Purpose: semantic / compositional recall, beyond keyword matching

### 3.3 Full object inventory

```
[table]   facts            记忆主表
[table]   facts_fts        FTS5 全文检索虚拟表
[table]   facts_fts_config / facts_fts_data / facts_fts_docsize / facts_fts_idx   FTS5 内部表
[table]   entities         实体表
[table]   fact_entities    记忆-实体关联表
[table]   memory_banks     按分类聚合的 HRR 记忆库向量
[table]   sqlite_sequence  自增序列(SQLite 内部)

[trigger] facts_ai / facts_au / facts_ad    facts 增删改时同步 FTS5 索引

[index]   idx_facts_category / idx_facts_trust / idx_entities_name   查询加速索引
[index]   sqlite_autoindex_*               UNIQUE 约束自动索引
```

### 3.4 Table creation SQL (the three key tables)

```sql
CREATE TABLE facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);

CREATE TABLE entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Verification

```bash
# 1) Confirm the provider is activated
hermes memory status
#   Expected: Provider: holographic / Plugin: installed ✓ / Status: available ✓

# 2) Confirm the database exists (requires at least one memory write first)
ls -la ~/.hermes/memory_store.db

# 3) Inspect the database contents directly (optional)
python3 - <<'PY'
import sqlite3
c = sqlite3.connect("/root/.hermes/memory_store.db")
print("facts:", c.execute("select count(*) from facts").fetchone()[0])
for r in c.execute("select fact_id, content, category from facts"):
    print(r)
PY
```

This investigation has been verified end-to-end: after `setup`, writing one test memory, `memory_store.db` is generated immediately (about 80 KB), with 10 tables + 3 triggers + indexes all in place.

For example, after entering the text below, inspect the database contents directly:

```Plain Text
立即调用 fact_store 工具，参数 action=add，category=reference，content 为下面引号内的原文。只执行工具调用，不要解释、不要复述、不要总结。

  content="llm-wiki 是把 Andrej Karpathy 的 LLM Wiki 模式做成 Hermes 技能：让 agent 维护一个纯 markdown 的互联知识库。与 RAG
  的区别：RAG 每次提问都从原始资料重新检索，跨文档关联要现场推；Wiki 把知识编译一次，页面已建好、交叉引用已连好、矛盾已标记、结论已综
  合，以后提问是读现成页面而非重新推导。分工：人负责选资料、定方向，agent
  负责摘要、建页、交叉引用、保持一致。它不是软件包，不需要编译，不依赖数据库或向量库，整个知识库就是一堆 .md 文件，可用 Obsidian、VS
  Code 或任何编辑器打开。"
```

Result:

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OWQ0ZmZlZmVmNjZjMGIyZWMzNWRiZGJkNWQ1MmI4MGRfYWNjOTAxOGRjMjZlNjNhMTUyNTJhMDEwNGE5Y2NkOGVfSUQ6NzcwNjA5NzY1MDU3OTg2ODY3MF8xNzk0MjE2Nzg0OjE3OTQzMDMxODRfVjM)

## One-sentence summary

Your memory structure = **one fact table** (with trust scores and usage counts, self-reinforcing) + **one layer of FTS5 keyword index** + **one layer of entity graph** + **one layer of HRR holographic vectors**, with three retrieval methods (keyword / entity association / semantic vector) coexisting, all packed into a single `~/.hermes/memory_store.db`. Installation is just one command: `hermes memory setup holographic`.
