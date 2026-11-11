#!/usr/bin/env python3
"""Generate bilingual command templates with a language model (plan item 1.4).

A template is one imperative sentence with exactly one ``{obj}`` and one ``{dest}``
placeholder, for example "could you move {obj} into {dest}?". ``tools/twin/sorting_task.py``
fills the placeholders with scene-unique references, so every (object, container) pair gets
as many phrasings as there are templates.

The script asks the model for templates per language (en, vi) and command kind (object,
class), keeps those that pass ``validate_template``, removes duplicates, and writes them with
the model name, the prompts and the date to ``configs/command_templates.json``.

  uv run python tools/data/generate_command_templates.py --per-kind 40
  uv run python tools/data/generate_command_templates.py --dry-run   # print the prompts only

The API key is read from GEMINI_API_KEY and never printed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "twin"))

DEFAULT_OUT = ROOT / "configs" / "command_templates.json"
KINDS = ("object", "class")
LANGUAGES = ("en", "vi")
MAX_ROUNDS = 4
VI_CHARS = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
PLACEHOLDER = re.compile(r"\{[^{}]*\}")

SYSTEM = (
    "You write instructions that a person would say to a robot arm sorting toy blocks into "
    "containers on a table. Answer with JSON only."
)

OBJECT_EXAMPLES = {
    "en": ("the red cylinder", "the small cube", "the blue object"),
    "vi": ("khối trụ đỏ", "khối lập phương nhỏ", "vật màu xanh dương"),
}
CLASS_EXAMPLES = {
    "en": ("a red object", "a cylinder", "a small object"),
    "vi": ("một vật màu đỏ", "một khối trụ", "một vật nhỏ"),
}
DEST_EXAMPLES = {
    "en": ("the left tray", "the round bin", "the grey tray"),
    "vi": ("khay bên trái", "thùng tròn", "khay xám"),
}
LANGUAGE_NAMES = {"en": "English", "vi": "Vietnamese with full diacritics"}

SCHEMA = {
    "type": "object",
    "properties": {"templates": {"type": "array", "items": {"type": "string"}}},
    "required": ["templates"],
}


def build_prompt(language: str, kind: str, count: int) -> str:
    objects = CLASS_EXAMPLES[language] if kind == "class" else OBJECT_EXAMPLES[language]
    meaning = (
        "any one object of a class (the robot may pick any matching object)"
        if kind == "class"
        else "one specific object"
    )
    return (
        f"Write {count} different instructions in {LANGUAGE_NAMES[language]} that tell the robot "
        f"to pick up {meaning} and put it into a container.\n"
        "Use the placeholder {obj} for the object phrase and {dest} for the container phrase, "
        "each exactly once. Do not use any other braces.\n"
        f"{{obj}} will be replaced by phrases such as: {', '.join(objects)}.\n"
        f"{{dest}} will be replaced by phrases such as: {', '.join(DEST_EXAMPLES[language])}.\n"
        "Vary the verb, the word order and the politeness: short commands, polite requests, "
        "casual speech. Each instruction has 3 to 20 words, one sentence, no numbering.\n"
        'Return {"templates": [...]}.'
    )


def validate_template(text: str, language: str) -> str | None:
    """Reason the template is unusable, or None if it is fine."""
    text = text.strip()
    found = PLACEHOLDER.findall(text)
    if sorted(found) != ["{dest}", "{obj}"]:
        return "placeholders"
    if text.find("{obj}") < 0 or text.count("{") != 2:
        return "placeholders"
    words = len(text.split())
    if not 3 <= words <= 20:
        return "length"
    letters = set(text.lower())
    if language == "vi" and not (letters & VI_CHARS):
        return "not vietnamese"
    if language == "en" and any(ord(ch) > 127 for ch in text):
        return "not english"
    if "\n" in text:
        return "multiline"
    return None


def normalise(text: str) -> str:
    return re.sub(r"[\s.!?,]+", " ", text.lower()).strip()


def collect(router, language: str, kind: str, target: int) -> tuple[list[str], list[str]]:
    """Valid unique templates and the models that produced them."""
    kept: dict[str, str] = {}
    models: list[str] = []
    for _ in range(MAX_ROUNDS):
        missing = target - len(kept)
        if missing <= 0:
            break
        result = router.call(SYSTEM, build_prompt(language, kind, missing + 10), SCHEMA)
        if result.payload is None:
            raise RuntimeError(f"no model answered: {result.error}")
        models.append(result.model)
        for raw in result.payload.get("templates", []):
            text = str(raw).strip().strip('"')
            if validate_template(text, language) is None:
                kept.setdefault(normalise(text), text)
    return list(kept.values())[:target], sorted(set(models))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-kind", type=int, default=40, help="templates per language and kind")
    parser.add_argument("--dry-run", action="store_true", help="print the prompts and exit")
    args = parser.parse_args(argv)
    prompts = {
        f"{lang}/{kind}": build_prompt(lang, kind, args.per_kind)
        for lang in LANGUAGES
        for kind in KINDS
    }
    if args.dry_run:
        for key, prompt in prompts.items():
            print(f"--- {key}\n{prompt}\n")
        return 0
    from twin_models import router_from_env

    router = router_from_env()
    router.escalation = ()
    templates: dict[str, dict[str, list[str]]] = {lang: {} for lang in LANGUAGES}
    models: set[str] = set()
    for lang in LANGUAGES:
        for kind in KINDS:
            kept, used = collect(router, lang, kind, args.per_kind)
            templates[lang][kind] = kept
            models.update(used)
            print(f"{lang}/{kind}: {len(kept)} templates")
    prompt_text = SYSTEM + "\n" + "\n".join(prompts.values())
    record = {
        "format": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "models": sorted(models),
        "system_prompt": SYSTEM,
        "prompts": prompts,
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "templates": templates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    short = [
        f"{lang}/{kind}"
        for lang in LANGUAGES
        for kind in KINDS
        if len(templates[lang][kind]) < args.per_kind
    ]
    print(f"wrote {args.out}" + (f"; below target: {', '.join(short)}" if short else ""))
    return 0 if not short else 2


if __name__ == "__main__":
    raise SystemExit(main())
