#!/usr/bin/env python3
"""Tokens per word of the held-out test commands under each candidate model's tokenizer.

A language model that splits Vietnamese words into many byte-level pieces sees longer, less
meaningful inputs than for English. This measurement is part of ADR 0005.

  python tools/train/tokenizer_stats.py configs/test_commands_v1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from pathlib import Path

MODELS = {
    "SmolVLA (SmolVLM2-500M)": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    "Evo-1 (InternVL3-1B)": "OpenGVLab/InternVL3-1B-hf",
}


def has_diacritics(text: str) -> bool:
    return any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", text))


def tokens_per_word(tokenizer, texts: list[str]) -> float | None:
    words = sum(len(t.split()) for t in texts)
    if not words:
        return None
    tokens = sum(len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts)
    return round(tokens / words, 2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "commands", type=Path, help="test command CSV (configs/test_commands_v1.csv)"
    )
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer

    with args.commands.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    result = {}
    for label, repo in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(repo)
        result[label] = {}
        for lang in ("en", "vi"):
            texts = [r["Command"] for r in rows if r["Language"] == lang]
            result[label][lang] = {
                "commands": len(texts),
                "tokens_per_word": tokens_per_word(tokenizer, texts),
                "tokens_per_word_with_diacritics": tokens_per_word(
                    tokenizer, [t for t in texts if has_diacritics(t)]
                ),
            }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
