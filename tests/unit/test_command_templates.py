"""Template validation (tools/data/generate_command_templates.py) and template banks."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "twin"))

import sorting_task as st  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "generate_command_templates", ROOT / "tools/data/generate_command_templates.py"
)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def test_validate_template() -> None:
    assert gen.validate_template("please move {obj} into {dest}", "en") is None
    assert gen.validate_template("làm ơn đặt {obj} vào {dest}", "vi") is None
    assert gen.validate_template("move {obj} to {dest} and {dest}", "en") == "placeholders"
    assert gen.validate_template("move {thing} to {dest}", "en") == "placeholders"
    assert gen.validate_template("{obj} {dest}", "en") == "length"
    assert gen.validate_template("dat {obj} vao {dest} nhe", "vi") == "not vietnamese"
    assert gen.validate_template("đặt {obj} vào {dest}", "en") == "not english"


def test_prompts_cover_languages_and_kinds() -> None:
    for lang in gen.LANGUAGES:
        for kind in gen.KINDS:
            prompt = gen.build_prompt(lang, kind, 40)
            assert "{obj}" in prompt and "{dest}" in prompt and "40" in prompt


class _Result:
    def __init__(self, payload):
        self.payload, self.model, self.error = payload, "fake-model", ""


class _Router:
    def __init__(self, batches):
        self.batches = list(batches)

    def call(self, system, prompt, schema):
        return _Result({"templates": self.batches.pop(0)})


def test_collect_filters_and_deduplicates() -> None:
    router = _Router(
        [
            ["put {obj} in {dest}", "Put {obj} in {dest}.", "bad {obj}", "move {obj} to {dest}"],
            ["could you place {obj} into {dest}", "put {obj} in {dest}"],
        ]
    )
    kept, models = gen.collect(router, "en", "object", 3)
    assert kept == [
        "put {obj} in {dest}",
        "move {obj} to {dest}",
        "could you place {obj} into {dest}",
    ]
    assert models == ["fake-model"]


def test_bank_templates_are_used(tmp_path) -> None:
    path = tmp_path / "bank.json"
    bank = {
        "en": {"object": ["kindly drop {obj} into {dest}"], "class": ["grab {obj}, then {dest}"]},
        "vi": {"object": ["nhờ bạn đặt {obj} vào {dest}"], "class": ["lấy {obj} cho vào {dest}"]},
    }
    path.write_text(json.dumps({"templates": bank}), encoding="utf-8")
    loaded = st.load_template_bank(path)
    rng = random.Random(0)
    scene = st.sample_scene(rng)
    texts = [
        st.make_command(scene, rng, "en", 0.0, bank=loaded, bank_share=1.0).text for _ in range(5)
    ]
    assert all(t.startswith("kindly drop the ") and " into the " in t for t in texts)
    vi = st.make_command(scene, rng, "vi", 0.0, bank=loaded, bank_share=1.0).text
    assert vi.startswith("nhờ bạn đặt ")
    assert st.load_template_bank(tmp_path / "missing.json") == {}
