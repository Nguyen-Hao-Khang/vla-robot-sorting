# W-C — Bilingual VLM / Semantic Evaluation Package

**Status:** Evaluation artifact implemented. Contract changes are proposals only.

**Date:** 2026-10-03

## Scope

This package evaluates the locked Sprint 1 boundary:

~~~text
natural-language command -> semantic reasoner -> validated structured intent
-> scene grounding -> 6-DoF grasp/planning/safety/execution
~~~

It does not generate robot motion or permit a VLM to issue motor commands. A model result must validate as structured intent before grounding, and the existing downstream safety/planning stack remains responsible for execution.

## Files and corpus design

| File | Purpose |
|---|---|
| experiments/corpora/bilingual_commands.jsonl | 66 stable cases, BC-001 through BC-066 |
| experiments/corpora/bilingual_commands.schema.json | Closed JSON Schema for every record |
| tests/scenarios/test_bilingual_corpus_contract.py | Offline parsing, schema-subset, integrity and coverage checks |

The corpus contains 29 English, 29 Vietnamese, and 8 EN/VI code-switching commands. Symmetric direct translation pairs use paired_case_id. Semantic-equivalence groups additionally share semantic_equivalence_id and normalized_intent_id. The test validator covers exactly the schema constructs used here, with no network dependency; a full JSON Schema validator may be added to CI later.

| Difficulty | Coverage | Expected behavior |
|---|---|---|
| L0 | Direct category/color/container | structured_success |
| L1 | Color, shape, size, location | structured_success |
| L2 | Multiple attributes and relation | structured_success |
| L3 | left/right, front/back, nearest/farthest, beside, between, on top | structured_success |
| L4 | Natural paraphrase, colloquial Vietnamese, reordered wording | structured_success |
| L5 | Valid code switching | structured_success |
| L6 | Multiple matches, pronouns, missing relation reference, stale/occluded references | clarify or reobserve |
| L7 | Contradictory or impossible requests | abort or reobserve |
| L8 | Unseen wording/category description and conflicting spatial language | clarify, reobserve, or abort |

Unseen wording changes the surface form of an existing semantic concept. Unseen concept/category wording has no licensed assumption of a unique visual mapping; it requires grounding evidence or clarification.

## Decision classes

| Class | Meaning | Required response |
|---|---|---|
| structured_success | Text provides a resolvable intent | Emit validated constraints, then ground |
| clarify | Several incompatible object/destination readings remain | Ask a bounded question; do not guess |
| reobserve | Current evidence is stale, occluded, or insufficient | Obtain a current observation revision |
| abort | Text is contradictory or impossible | Return typed semantic refusal before planning |

The corpus labels target and destination grounding independently as unique, multiple, not visible, not present, or not applicable. Cases tagged proposed_policy are review proposals, not assertions that a new physical safety policy is already implemented.

## Scoring and failure attribution

Report denominators and exclusions for every metric. Do not report a single headline accuracy.

| Metric | Denominator | Correct result |
|---|---|---|
| Behavior accuracy | All evaluated cases | Exact decision class |
| Intent exact match | Gold structured-success cases | Operation and normalized intent ID match |
| Field precision/recall/F1 | Applicable gold target/destination attributes and relations | Correct normalized fields; macro and micro reported |
| Clarify/reobserve/abort accuracy | Gold cases of that class | Correct safe decision |
| Target/destination grounding accuracy | Semantically correct, visible, unambiguous success cases | Correct scene instance |
| Relation grounding accuracy | Visible unambiguous relation cases | Correct instance from relation/reference |
| Bilingual consistency | Semantic-equivalence groups | Same behavior, normalized intent, and constraints |
| End-to-end success | Only semantically and visually correct executed episodes | Safe intended placement; retain downstream code |

Failure attribution is strict:

1. Wrong behavior, operation, attribute, relation, or bilingual intent is semantic parsing failure.
2. Correct semantic constraints but wrong target/destination instance is visual grounding failure.
3. Correct semantic and grounding decisions followed by grasp, IK, collision, trajectory, or execution failure is downstream failure. It is not a VLM failure.

Clarify, reobserve, and abort cases are excluded from manipulation and instance-grounding denominators because their correct outcome is to avoid unsafe guessing.

## Proposed contract changes for Tech Lead review

The existing SkillCommand correctly keeps semantic decisions away from actuator control, but its open semantic_constraints dictionary cannot by itself preserve enough evaluation provenance. W-C does not edit production contracts.

| Proposed field | Need and semantics | Blocking | Compatibility impact |
|---|---|---|---|
| decision enum | structured_success, clarify, reobserve, abort | Blocking for safe semantic evaluation | Additive wrapper preferred |
| normalized_intent_id | Stable meaning across EN/VI surface forms | Blocking for bilingual metric | Additive |
| source_language | en, vi, code_switch | Optional | Additive |
| semantic_confidence | Telemetry confidence in [0,1], never a substitute for ambiguity rules | Optional | Additive |
| ambiguity object | Reason code and clarification slots | Blocking for clarify | Additive structured data |
| grounding_constraints | Category, attributes, relation and reference before instance selection | Blocking for reproducible grounding evaluation | Additive; can mirror current dictionary temporarily |
| observation_revision | Binds an intent to the real scene/transform revision | Blocking for stale-reference evidence | Thread from observation, never invented by VLM |
| provenance object | Model ID, prompt/config hash, case ID, timestamps; no secrets | Optional runtime, blocking reports | Additive |

A separate semantic-result wrapper is preferable to enlarging SkillCommand with every evaluation field. SkillCommand remains a high-level command and never a motor interface.

## Integration protocol

Freeze corpus hash, model ID, prompt hash, scene/observation revision, and result configuration. Validate a model output before converting it to any runtime object. Score language first, then grounding, then grasp/planning/execution. Preserve malformed structured output as a semantic failure record. Archive per-case results and confusion matrices by behavior, difficulty, language and equivalence group.

The next recommended integration step is a project-owned semantic-result wrapper and deterministic fake grounding fixture that consumes this corpus without editing simulator, motion, planning, control, or safety code.
