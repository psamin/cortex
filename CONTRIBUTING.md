# Contributing to Cortex

Several people, each driving coding agents, work in this repo at once. These rules keep work from
overlapping and merges clean. They apply equally to humans and AI agents. **If you are an agent, read
this file and the Claims table in [docs/ROADMAP.md](docs/ROADMAP.md) before you edit anything.**

## Workflow: issue → claim → branch → PR

1. **Sync first.** Run `git fetch && git switch main && git pull --ff-only`. Rebase your branch on
   `main` before every push.
2. **Check for overlap.** If an open issue, or a row in the ROADMAP Claims table, lists a file you need,
   coordinate on that issue first. Otherwise wait until it merges.
3. **Open an issue** for the package. Give the goal, the **files touched** (`[new]`, `[mod]` or `[del]`,
   full paths), any interfaces others depend on, a "done when" check someone else can run, and
   what it depends on or blocks.
4. **Claim it.** Assign yourself, add a row to the Claims table, and name the driver. For agents,
   include the human driving them, e.g. "Claude Code, driven by @AneeshD04".
5. **Branch.** Name it `<issue#>-<slug>`. Until issues exist, use `feat/<slug>`, `fix/<slug>` or
   `docs/<slug>`.
6. **Stay inside your declared files.** If you need another one, update the issue and the claim
   first.
7. **Open a PR** whose body says `Closes #<n>` and lists the files touched and how you verified the
   change. Keep it under about 400 changed lines, and split bigger work. Stacked PRs say what they
   are stacked on.
8. **Check before review.** Run `cd backend && uv run ruff check cortex tests && uv run pytest` and,
   for MCP changes, `cd mcp && npx tsc --noEmit -p .`. Post the results in the PR.
9. **Merge.** Squash-merge, delete the branch, and remove your Claims row.

## Hot files

These are shared by everyone, so change them in small, fast-merging PRs:

- `backend/cortex/models.py`
- `backend/cortex/services/agent_orchestrator.py` (the tool registry): add tools; do not reorder
  them
- `backend/cortex/api/app.py`
- `mcp/src/tools.ts`
- `backend/pyproject.toml` and `backend/uv.lock`: add dependencies in their own PR
- `README.md`, `ARCHITECTURE.md`, this file

## Conventions

- **Python 3.12 with `uv`.** Ruff, line length 120. Type hints on public functions. Validate external
  data (agent input, LLM output) with Pydantic.
- **Explicit seeds.** Every source of randomness takes a seed.
- **Tests for every new module.** Sim tests use the real pinned EnvHub model.
- **Windows is a supported dev platform.** Don't use `os.uname`. Create, render and close MuJoCo
  renderers on one thread.
- **Commits end with attribution lines** when an agent wrote them.
- **No silent substitution.** A provider that is missing is reported, never replaced by another
  provider; demos keep their provenance; the scripted oracle is always labelled.
