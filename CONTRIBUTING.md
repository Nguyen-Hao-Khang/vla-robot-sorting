# Contributing

This guide covers branches, commits, code and pull requests.

## 1. Branches

`main` always passes lint and tests. Work happens on short-lived branches that reach `main` through a pull request. Direct pushes to `main` are blocked.

Branch name: `<owner>/<topic>`

| Part | Rule |
|---|---|
| `owner` | Member handle: `khangnh`, `huyvq`, `phucpg` |
| `topic` | 2–5 lowercase English words joined by hyphens, ASCII only |

Examples: `khangnh/scene-randomizer`, `huyvq/smolvla-benchmark`, `phucpg/instruction-generator`.

## 2. Commits

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <summary>

<body: what changed and why, wrapped at 72 columns>

Refs: #<issue>
```

- `summary`: imperative mood, lowercase, no trailing period, at most 50 characters.
- `body`: optional; explain the reason, not the diff.
- One logical change per commit. Lint and tests pass after every commit.

| `type` | Use for |
|---|---|
| `feat` | New functionality |
| `fix` | Bug fix |
| `test` | Tests only |
| `docs` | Documentation only |
| `exp` | Experiment configuration, evaluation results, result tables |
| `build` | Dependencies, CI, tooling |
| `refactor` | Restructuring without behaviour change |

Common scopes: `sim`, `data`, `train`, `eval`, `robot`, `safety`.

```
feat(data): randomize object layout per episode
fix(sim): add table objects to the curobo world
exp(eval): add smolvla simulation results
build(ci): cache uv environment
```

## 3. Code

- Format and lint with `ruff` (line length 100). `uv run pre-commit install` runs the checks on every commit.
- Write identifiers, comments, docstrings and log messages in English. Comments state intent or constraints, not what the code already says.
- Add a test for every behaviour change. Unit tests must run without a GPU, the simulator or network access; mark exceptions with `@pytest.mark.simulator` or `@pytest.mark.network`.
- Never relax a safety threshold to make a run pass.
- No absolute paths from a personal machine. Use command-line arguments or environment variables.
- Planned work: `TODO(<scope>): <action>`.

## 4. Pull requests

Before opening a pull request:

```sh
uv run ruff check .
uv run pytest
```

1. Update `main` and branch from it.
2. Commit following section 2 and push the branch.
3. Open a pull request into `main`. Use a Conventional Commits title, describe what changed and how you tested it, and add `Closes #<issue>`.
4. Check your own diff. When CI is green, merge with **Squash and merge**. No approval is required.
5. GitHub deletes the branch after the merge.

## 5. Data and results

Keep these out of Git:

- API keys, tokens, `.env` files;
- datasets, checkpoints, videos, training logs: store them on the Hugging Face Hub or Kaggle and link them from the pull request;
- build outputs.

Every reported number must trace to a commit, a configuration, the scene seeds and the raw evaluation log ([evaluation protocol](docs/evaluation-protocol.md)).

## 6. Issues

Use the issue forms: **Bug report** for defects, **Feature request** for new capabilities, **Task** for a work item from the [project plan](docs/project-plan.md).
