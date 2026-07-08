#!/usr/bin/env python3
"""
FlagOS Lab — Continuous Chat Client
OpenAI-compatible multi-turn conversation client, runs until the token limit is reached.
Usage: python chat_client.py [--base-url URL] [--model NAME] [--max-tokens N] [--max-context N]
"""

import argparse
import sys
from openai import OpenAI


def count_tokens_approx(text: str) -> int:
    """Rough token estimate (~1.5 tokens/CJK char, ~1.3 tokens/EN word)"""
    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en_words = len(text.split()) - cn_chars // 2
    return int(cn_chars * 1.5 + max(en_words, 0) * 1.3)


def build_parser():
    p = argparse.ArgumentParser(description="FlagOS Lab Continuous Chat Client")
    p.add_argument("--base-url", default="http://localhost:30000/v1",
                    help="vLLM service URL (default: http://localhost:30000/v1)")
    p.add_argument("--model", default="qwen", help="Model name (default: qwen)")
    p.add_argument("--max-tokens", type=int, default=512,
                    help="Max tokens per response (default: 512)")
    p.add_argument("--max-context", type=int, default=4096,
                    help="Total context window token limit (default: 4096)")
    p.add_argument("--temperature", type=float, default=0.7,
                    help="Generation temperature (default: 0.7)")
    p.add_argument("--system", default="You are a helpful assistant.",
                    help="System prompt")
    return p


def main():
    args = build_parser().parse_args()

    client = OpenAI(base_url=args.base_url, api_key="any")
    messages = [{"role": "system", "content": args.system}]
    total_tokens_used = 0

    print("=" * 60)
    print(f"  FlagOS Lab Continuous Chat Client")
    print(f"  Model: {args.model} | Context limit: {args.max_context} tokens")
    print(f"  Service URL: {args.base_url}")
    print(f"  /quit to exit | /clear to reset | /tokens to check usage")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("\033[92mYou: \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Command handling
        if user_input == "/quit":
            print("Goodbye!")
            break
        elif user_input == "/clear":
            messages = [{"role": "system", "content": args.system}]
            total_tokens_used = 0
            print("\033[93m[History cleared]\033[0m\n")
            continue
        elif user_input == "/tokens":
            print(f"\033[93m[Used ~{total_tokens_used} / {args.max_context} tokens]\033[0m\n")
            continue

        messages.append({"role": "user", "content": user_input})

        # Check if approaching context limit
        estimated = sum(count_tokens_approx(m["content"]) for m in messages)
        if estimated + args.max_tokens > args.max_context:
            print(f"\033[91m[Warning] Approaching context limit (~{estimated} tokens), trimming earlier messages...\033[0m")
            # Keep system + most recent turns
            while len(messages) > 3 and estimated + args.max_tokens > args.max_context:
                messages.pop(1)  # Remove earliest user/assistant message
                estimated = sum(count_tokens_approx(m["content"]) for m in messages)
            if estimated + args.max_tokens > args.max_context:
                print("\033[91m[Error] Still exceeds limit after trimming, use /clear to start over\033[0m\n")
                messages.pop()  # Undo the just-added user message
                continue

        try:
            print(f"\033[94mQwen: \033[0m", end="", flush=True)
            # Try streaming output
            try:
                stream = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    stream=True,
                )
                assistant_reply = ""
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        assistant_reply += content
                        print(content, end="", flush=True)
                print()
            except Exception:
                # Fall back to non-streaming
                response = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                assistant_reply = response.choices[0].message.content
                print(assistant_reply)

                if hasattr(response, "usage") and response.usage:
                    total_tokens_used = response.usage.total_tokens

            messages.append({"role": "assistant", "content": assistant_reply})
            # Update token estimate
            total_tokens_used = sum(count_tokens_approx(m["content"]) for m in messages)
            print(f"\033[90m[~{total_tokens_used}/{args.max_context} tokens]\033[0m\n")

        except Exception as e:
            print(f"\n\033[91m[Request failed] {e}\033[0m\n")
            messages.pop()  # Undo the failed user message


if __name__ == "__main__":
    main()
