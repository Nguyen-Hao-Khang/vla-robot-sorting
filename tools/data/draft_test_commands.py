#!/usr/bin/env python3
"""Draft the held-out test commands for team review (plan item 1.5).

Each row belongs to one evaluation episode: the episode seed fixes the scene, the allowed
objects and the destination (``twin_eval.plan_episode``), and a language model drafts a free
command for it in a given style. The team then reads every row, edits the wording and marks
it reviewed. ``twin_eval.py run --command-source human`` runs exactly these episodes.

The output is a CSV laid out like a Google Forms export (Timestamp first, one column per
question) so it can be opened in a spreadsheet and edited there.

  uv run python tools/data/draft_test_commands.py --seed 9 --count 160 \
      --out configs/test_commands_v1.csv

Drafts that reproduce a training template exactly are rejected. The API key is read from
GEMINI_API_KEY and never printed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "twin"))

import sorting_task as st  # noqa: E402
import twin_eval as te  # noqa: E402

COLUMNS = (
    "Timestamp",
    "Scene seed",
    "Language",
    "Command kind",
    "Scene",
    "Allowed objects",
    "Destination",
    "Command",
    "Style",
    "Reviewed",
    "Notes",
)
STYLES = {
    "en": (
        "short imperative",
        "polite request",
        "casual speech",
        "describe the object with more attributes than needed",
        "use a synonym for the shape or the container",
        "put the destination first",
        "a question asking the robot to do it",
        "explain it as part of tidying up the table",
    ),
    "vi": (
        "câu mệnh lệnh ngắn",
        "lời nhờ lịch sự",
        "giọng nói chuyện thân mật, có từ địa phương",
        "mô tả vật bằng nhiều thuộc tính hơn cần thiết",
        "dùng từ đồng nghĩa cho hình dạng hoặc khay",
        "nói khay đích trước",
        "câu hỏi nhờ robot làm",
        "gõ không dấu như khi nhắn tin vội",
    ),
}
SYSTEM = (
    "You write realistic instructions that different people would give to a robot arm that "
    "sorts toy blocks into containers on a table. Answer with JSON only."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "commands": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["id", "text"],
            },
        }
    },
    "required": ["commands"],
}
VI_CHARS = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
WORD = re.compile(r"[^\W_]+", re.UNICODE)


def scene_text(scene: st.Scene) -> str:
    objects = "; ".join(o.name for o in scene.objects)
    containers = "; ".join(
        f"{c.position}: {c.colour} {c.outline} {c.depth}" for c in scene.containers
    )
    return f"Objects: {objects}. Containers: {containers}."


def draft_items(seed: int, count: int, class_share: float) -> list[dict]:
    """One item per episode; languages alternate so both get the same number of rows."""
    rng = random.Random(f"{seed}/styles")
    items = []
    for index in range(count):
        language = ("vi", "en")[index % 2]
        plan = te.plan_episode(te.episode_seed(seed, index), language, class_share, {}, 0.0)
        command = plan.command
        dest = plan.scene.container(command.destination)
        definite = command.kind == "object"
        items.append(
            {
                "id": index,
                "seed": plan.seed,
                "language": language,
                "kind": command.kind,
                "scene": scene_text(plan.scene),
                "targets": list(command.targets),
                "destination": command.destination,
                "object_phrase": st.describe_object(command.object_reference, language, definite),
                "container_phrase": st.describe_container(
                    command.container_reference, dest, language
                ),
                "style": rng.choice(STYLES[language]),
            }
        )
    return items


def build_prompt(batch: list[dict]) -> str:
    lines = [
        "For each item, write one instruction in the given language and style.",
        "The instruction must identify the object and the container so that a person looking "
        "at the scene knows exactly what is meant. You may rephrase the given phrases, add "
        "attributes or use synonyms, but never name a different object or container.",
        "For kind 'class', any object of that class is acceptable: say 'a ...' / 'một ...', "
        "not 'the ...'.",
        "Vietnamese must have full diacritics unless the style says to type without them.",
        "Each instruction is one sentence of 3 to 25 words.",
        'Return {"commands": [{"id": <id>, "text": "..."}]}.',
        "",
    ]
    for item in batch:
        lines.append(
            json.dumps(
                {
                    k: item[k]
                    for k in (
                        "id",
                        "language",
                        "kind",
                        "scene",
                        "object_phrase",
                        "container_phrase",
                        "style",
                    )
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def normalise(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def template_texts(item: dict, bank: dict) -> set[str]:
    """Every training command that the templates produce for this item's phrases."""
    lang, kind = item["language"], item["kind"]
    builtin = {
        ("en", "object"): st.EN_OBJECT_TEMPLATES,
        ("en", "class"): st.EN_CLASS_TEMPLATES,
        ("vi", "object"): st.VI_OBJECT_TEMPLATES,
        ("vi", "class"): st.VI_CLASS_TEMPLATES,
    }[(lang, kind)]
    obj, dest = item["object_phrase"], item["container_phrase"]
    texts = set()
    for template in (*builtin, *bank.get(lang, {}).get(kind, [])):
        for o in {obj, obj.removeprefix("the ")}:
            for d in {dest, f"the {dest}"}:
                texts.add(normalise(template.format(obj=o, dest=d)))
    return texts


def check_draft(text: str, item: dict, bank: dict) -> str | None:
    words = len(text.split())
    if not 3 <= words <= 25:
        return "length"
    if "\n" in text or "{" in text:
        return "format"
    if item["language"] == "en" and any(ord(ch) > 127 for ch in text):
        return "not english"
    no_marks = "không dấu" in item["style"]
    has_marks = any(ch in VI_CHARS for ch in text.lower())
    if item["language"] == "vi" and not no_marks and not has_marks:
        return "missing diacritics"
    if normalise(text) in template_texts(item, bank):
        return "same as a training template"
    return None


def write_csv(path: Path, items: list[dict], drafts: dict[int, str]) -> None:
    stamp = dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(COLUMNS)
        for item in items:
            writer.writerow(
                [
                    stamp,
                    item["seed"],
                    item["language"],
                    item["kind"],
                    item["scene"],
                    " | ".join(item["targets"]),
                    item["destination"],
                    drafts.get(item["id"], ""),
                    item["style"],
                    "no",
                    "",
                ]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=9, help="evaluation run seed (held out)")
    parser.add_argument("--count", type=int, default=160)
    parser.add_argument("--class-share", type=float, default=0.3)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--out", type=Path, default=ROOT / "configs" / "test_commands_v1.csv")
    parser.add_argument(
        "--templates", type=Path, default=ROOT / "configs" / "command_templates.json"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the first prompt and exit")
    args = parser.parse_args(argv)
    items = draft_items(args.seed, args.count, args.class_share)
    if args.dry_run:
        print(build_prompt(items[: args.batch]))
        return 0
    from twin_models import router_from_env

    router = router_from_env()
    router.escalation = ()
    bank = st.load_template_bank(args.templates)
    drafts: dict[int, str] = {}
    pending = list(items)
    for _round in range(3):
        retry = []
        for start in range(0, len(pending), args.batch):
            batch = pending[start : start + args.batch]
            result = router.call(SYSTEM, build_prompt(batch), SCHEMA)
            answers = {}
            if result.payload is not None:
                answers = {int(c["id"]): str(c["text"]).strip() for c in result.payload["commands"]}
            for item in batch:
                text = answers.get(item["id"], "")
                if text and check_draft(text, item, bank) is None:
                    drafts[item["id"]] = text
                else:
                    retry.append(item)
        pending = retry
        if not pending:
            break
    write_csv(args.out, items, drafts)
    print(f"wrote {args.out}: {len(drafts)}/{len(items)} drafted, {len(pending)} left empty")
    return 0 if not pending else 2


if __name__ == "__main__":
    raise SystemExit(main())
