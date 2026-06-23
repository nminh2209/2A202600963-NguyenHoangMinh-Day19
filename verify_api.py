#!/usr/bin/env python3
"""Verify OpenAI API is connected and report recent pipeline token usage."""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.config import COST_REPORT_PATH, LLM_MODEL, get_openai_api_key

load_dotenv()


def main():
    key = get_openai_api_key()
    print("=" * 50)
    print("OpenAI API Verification")
    print("=" * 50)

    if not key:
        print("STATUS: DEMO MODE (no API key in .env)")
        return

    print(f"STATUS: Real LLM mode")
    print(f"Model:  {LLM_MODEL}")
    print(f"Key:    sk-...{key[-8:]}")

    client = OpenAI(api_key=key)
    r = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": "Reply with exactly: LLM_CONNECTED"}],
        max_tokens=10,
    )
    print(f"Test response: {r.choices[0].message.content}")
    print(f"Test tokens:   {r.usage.total_tokens}")
    print(f"Model used:    {r.model}")
    print()
    print("If dashboard shows $0.00, usage may be delayed 5-15 min")
    print("or cost is very small (~$0.01 for this lab).")
    print()

    if COST_REPORT_PATH.exists():
        cost = json.loads(COST_REPORT_PATH.read_text(encoding="utf-8"))
        print("Last pipeline run:")
        print(f"  Mode: {cost.get('mode')}")
        idx = cost.get("indexing", {})
        ev = cost.get("evaluation", {})
        print(f"  Indexing tokens: {idx.get('tokens', 0)}")
        print(f"  Eval Flat tokens: {ev.get('total_tokens_flat', 0)}")
        print(f"  Eval Graph tokens: {ev.get('total_tokens_graph', 0)}")
        total = idx.get("tokens", 0) + ev.get("total_tokens_flat", 0) + ev.get("total_tokens_graph", 0)
        print(f"  Total logged: ~{total} tokens")


if __name__ == "__main__":
    main()
