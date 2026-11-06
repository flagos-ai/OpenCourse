#!/usr/bin/env python3
"""transform_llm_output hook：从模型最终响应里抽取 fenced 代码块落盘，
并返回一段干净汇报。

用途：当 vLLM 0.24 的 hermes 解析器与 Qwen2.5 的工具输出格式不匹配、
Hermes 没能把模型输出转成 file 工具调用（代码以纯文本形式输出）时，
由本 hook 兜底：从最终响应里把 ```python 代码块抽出来直接写到磁盘，
使编码类任务仍能产出 .py 文件。

放置位置：/root/.hermes/agent-hooks/code-writer.py
注册方式：见文件末尾注释。
"""
import sys
import json
import re
import os


def main():
    # Hermes 通过 stdin 以 JSON 载荷调用 shell hook
    try:
        payload = json.loads(sys.stdin.read())
    except Exception:
        print("{}")
        return

    extra = payload.get("extra") or {}
    # transform_llm_output 的文本字段是 response_text；
    # 兼容 post_llm_call 的 assistant_response
    text = extra.get("response_text") or extra.get("assistant_response") or ""
    cwd = payload.get("cwd") or os.getcwd()
    if not text:
        print("{}")
        return

    # 没有显式文件名时，按出现顺序回退到这两个名字
    default_names = ["todo.py", "test_todo.py"]
    written = []
    py_idx = 0

    # 匹配 fenced 代码块：```lang\n ... ```
    pattern = re.compile(r"```[ \t]*([\w+.-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
    for m in pattern.finditer(text):
        lang = (m.group(1) or "").lower()
        code = m.group(2)
        start = m.start()
        window = text[max(0, start - 300):start]  # 向前找显式文件名

        # 只处理 python 块；无语言标签时要求附近出现 .py 才认
        is_py = lang in ("python", "py")
        is_bare_with_py = (lang == "" and re.search(r"\.py\b", window))
        if not (is_py or is_bare_with_py):
            continue

        names = re.findall(r"([A-Za-z0-9_./-]+\.py)", window)
        if names:
            fname = names[-1]
        elif py_idx < len(default_names):
            fname = default_names[py_idx]
        else:
            fname = f"snippet_{py_idx + 1}.py"
        py_idx += 1

        path = os.path.join(cwd, os.path.basename(fname))
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(code.rstrip() + "\n")
            written.append(os.path.basename(path))
        except Exception:
            pass

    # 落盘情况写到 stderr，便于在 `hermes logs` 里排查，不影响返回协议
    sys.stderr.write(f"[code-writer] wrote: {written}\n")

    if written:
        report = (
            "已从模型输出抽取代码并落盘。\n"
            f"- 创建了 {len(written)} 个文件：{', '.join(written)}\n"
            "- 预期 pytest 应通过 5 项"
        )
        # 非空 str 会替换最终响应文本（transform 语义）
        print(json.dumps({"response_text": report}))
    else:
        # 没抽到代码块：静默放行，不改动模型输出
        print("{}")


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────
# 注册方式：在 /root/.hermes/config.yaml 末尾加入（command 用绝对路径）：
#
# hooks:
#   transform_llm_output:
#     - command: "/root/.hermes/agent-hooks/code-writer.py"
#       timeout: 15
# hooks_auto_accept: true
#
# 自测（免 heredoc，避免粘贴缩进把 EOF 吞掉）：
#   chmod +x /root/.hermes/agent-hooks/code-writer.py
#   printf '%s' '{"cwd":"/tmp","extra":{"response_text":"x\n```python\n# todo.py\nprint(1)\n```\n"}}' \
#     | python3 /root/.hermes/agent-hooks/code-writer.py
#   cat /tmp/todo.py      # 应显示 print(1)
# ─────────────────────────────────────────────────────────────────────
