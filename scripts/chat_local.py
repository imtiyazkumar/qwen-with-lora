"""Chat with the selected local model. Use /clear or /quit."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.inference.runtime import LocalModel  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    runtime = LocalModel()
    history: list[dict[str, str]] = []
    print(
        "Local Qwen3.5-2B. General model knowledge; RAG is not connected yet."
    )
    print("Commands: /clear, /quit. Answers are limited to 128 output tokens.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question == "/quit":
            return
        if question == "/clear":
            history.clear()
            print("Conversation cleared.")
            continue
        if not question:
            continue
        candidate = history + [{"role": "user", "content": question}]
        try:
            result = runtime.generate(candidate)
        except ValueError as exc:
            print(str(exc))
            continue
        print("\nAssistant:", result["answer"])
        if result["output_truncated"]:
            print("[Output token limit reached.]")
        history = candidate + [
            {"role": "assistant", "content": result["answer"]}
        ]


if __name__ == "__main__":
    main()
