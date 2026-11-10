# Contributing

These rules apply to all three team members.

## Workflow

1. Pick or create an issue on the project board. Each task in the project plan maps to one issue.
2. Create a branch from the latest `main`:

   ```bash
   git switch main && git pull
   git switch -c <name>/<task>
   ```

   Branch names use lowercase and hyphens: `<name>` is the author's first name and `<task>` is a short description.
   Examples: `khang/scene-randomizer`, `huy/smolvla-bakeoff`, `phuc/task-spec`.
3. Commit in small, complete steps. Each commit must leave tests and lint passing.
4. Push the branch and open a pull request into `main`. Write `Closes #<issue>` in the description.
5. Another member reviews the pull request. Authors do not merge their own pull requests.
6. Merge with **Squash and merge**. GitHub deletes the branch after the merge.

Nobody pushes directly to `main`.

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <summary>

<optional body: what changed and why>

Refs #<issue>
```

| Type | Use for |
|---|---|
| `feat` | New capability |
| `fix` | Bug fix |
| `docs` | Documentation or report text |
| `test` | Tests only |
| `refactor` | Code change with no behaviour change |
| `exp` | Experiment configs, evaluation runs, result tables |
| `chore` | Tooling, dependencies, repository settings |

Scopes: `sim`, `twin`, `data`, `train`, `eval`, `robot`, `safety`, `perception`, `semantics`, `report`, `docs`, `ci`.

Rules:
- Write the summary in English, imperative mood, lowercase, under 72 characters, with no trailing period.
  - Good: `feat(data): add randomized scene generator`.
  - Avoid: `Update files`, `fix bug`, `WIP`.
- Use the body to explain why. The diff already shows what changed.
- The squash commit on `main` keeps the pull request title, so the PR title follows the same format.

## Code

- **Language.** Code, comments, identifiers, logs and commit messages are in English. The report in `report/` is in Vietnamese.
- **Python.** Use Python 3.10 to 3.12. Format and lint with `ruff` (line length 100). Run `uv run ruff check .` and `uv run pytest` before you push.
- **Tests.** Add or update tests with every behaviour change. Tests must run without a simulator, a GPU or network access.
- **Interfaces.** Third-party libraries stay behind the contracts in `src/vla_robot_sorting/contracts` (see ADR-0001).
- **Safety.** Never loosen a safety limit to make a run pass.
- **Ground truth.** Label simulator ground truth as ground truth. Never report it as perception.

## What does not go into the repository

- API keys, tokens and `.env` files. Keys live in environment variables only.
- Datasets, checkpoints, videos and training outputs. Store them on Hugging Face Hub or Kaggle and put the link in the pull request.
- Personal notes, meeting chat logs and drafts unrelated to the code or report.
- Generated files: LaTeX build output, `__pycache__`, notebook outputs.

## Results

A result goes into the report only if the repository can trace it to:

- the commit that produced it;
- the config and seed;
- the raw log or evaluation file, stored outside Git and linked from the pull request.

## Report

The report lives in `report/` (LaTeX, Vietnamese). Build it with `latexmk -pdf main.tex` inside `report/`. Commit only `.tex`, `.bib` and figure sources, never the PDF.
