# Contributing to DreamCache

DreamCache asks one research question: **can selective speculative rollouts,
triggered only on hard decisions, match always-on search success at a fraction
of the compute?** Several people, each driving coding agents, work in this repo
at once. These rules exist so that work doesn't overlap, merges stay clean, and
every number in the report can be trusted.

The rules apply equally to humans and AI agents. **If you are an AI agent, read
this whole file before you touch anything, and follow it literally.**

---

## 1. The workflow: issue → claim → branch → PR

Every change, however small, follows these steps.

1. **Check for overlap first.**
   ```bash
   gh issue list --state open --label "status:in-progress"
   gh issue list --state open --search "<file or module name>"
   ```
   If an open issue already lists a file you need under **Files touched**,
   you may not edit that file. Comment on that issue to coordinate, or wait for
   it to merge.

2. **Open an issue** with the *Task* template (`.github/ISSUE_TEMPLATE/task.yml`).
   It must include:
   - **Goal.** One or two sentences.
   - **Files touched.** Every file you will create, modify, or delete, each
     tagged `[new]`, `[mod]`, or `[del]`. Be exact, using full paths. Globs are
     allowed only for new files inside a directory you own, e.g.
     `[new] src/dreamcache/detection/*.py`.
   - **Interfaces.** Any public function, class, config key, or file format
     that others will depend on, or that you are changing.
   - **Done when.** A check someone else can run, e.g. "`pytest tests/test_cloning.py` passes".
   - **Depends on / blocks.** Other issue numbers.

3. **Claim it.** Assign yourself and add the label `status:in-progress`. One
   person or agent per issue. Agents put their driver's name in the issue,
   e.g. "agent run by @alice".

4. **Branch** from up-to-date `main`, named `<issue#>-<short-slug>`:
   ```bash
   git switch main && git pull --ff-only
   git switch -c 12-clone-validation
   ```

5. **Stay inside your declared files.** If you find you need a file you
   didn't list, **edit the issue first** to add it, then check step 1 again
   for that file. Do not quietly widen your scope.

6. **Open a PR** using the PR template. Its body includes `Closes #<n>`. Keep
   PRs small (under ~400 changed lines when possible). Split big work into
   several issues.

7. **Before you request review**, rebase on `main`, run `make check` (lint and
   tests), and make sure the diff touches only the files declared in the issue.

8. **Merge.** At least one approving review is required, and the author does
   not merge their own PR unless the reviewer says to. Squash-merge. Delete
   the branch afterwards.

Issues that sit in `status:in-progress` for more than 12 hours with no push may
be released. Ping the claimant first.

---

## 2. Module ownership map

The codebase is split so that independent work streams touch disjoint
directories. Each issue should, wherever possible, stay inside **one** row of
this table.

| Area | Paths | Notes |
|---|---|---|
| Core interfaces | `src/dreamcache/types.py`, `src/dreamcache/config.py` | **Frozen contract.** Changes need a dedicated issue labeled `area:interfaces` and a review from one other teammate. |
| Environments & cloning | `src/dreamcache/environments/` | Clone correctness is load-bearing. Every change needs tests. |
| Policies (LLM, heuristic, candidates) | `src/dreamcache/policies/` | Ollama client lives here. |
| Difficulty detection | `src/dreamcache/detection/` | Observation-only. See §5. |
| Speculation engine | `src/dreamcache/speculation/` | Rollouts, evaluator, parallelism. |
| Agents (A–I conditions) | `src/dreamcache/agents/` | One file per condition. |
| Experiments & stats | `src/dreamcache/experiments/` | Runner, metrics, bootstrap. |
| Analysis & plots | `analysis/`, `report/` | Read results only. Never write to `results/`. |
| Configs | `configs/` | One file per experiment. Add new files rather than editing shared ones. |
| Cluster & scripts | `scripts/`, `slurm/` | See §6. |
| Tests | `tests/` | Mirror the source path: `tests/<area>/test_<module>.py`. |

### Hot files (conflict magnets)

Edit these only through a dedicated, short-lived issue, and merge it fast:

- `pyproject.toml` and `uv.lock`. To add a dependency, open an issue titled
  `deps: add <pkg>`. Never regenerate the lock file inside a feature PR.
- `README.md`. Keep it short. Detail goes in `docs/<topic>.md`.
- `src/dreamcache/__init__.py`, and any registry that maps names to classes.
  Register new components by **adding one line**, never by reordering.
- `CONTRIBUTING.md` and `.github/`.

---

## 3. Interfaces first

When two work streams meet (for example, the speculation engine consumes
candidate actions from policies), the shared types land **first**, in a small
interfaces PR, before either side builds on them. After that, each side codes
against the types and can be tested with stubs. Don't change a shared signature
inside a feature PR.

---

## 4. Code conventions

- Python 3.11+. Manage the environment with `uv` (`uv sync`, `uv run pytest`).
- Format and lint with `ruff`. Type hints on public functions.
  Validate external data (LLM output, configs) with Pydantic.
- No hidden global state. Every source of randomness takes an explicit seed
  or `np.random.Generator`.
- Every new module comes with tests. Anything slow is marked
  `@pytest.mark.slow`, and any test that needs a live Ollama server is marked
  `@pytest.mark.ollama`. Plain `pytest` must pass on a laptop CPU, with no
  Ollama running, in under a minute.
- Match the style of the surrounding code. Don't refactor unrelated code in a
  feature PR. If you spot an unrelated problem, open an issue for it.

---

## 5. Research integrity rules (non-negotiable)

These rules protect the validity of the results. A PR that breaks one is
rejected, however good the code is.

1. **Never fabricate or hand-edit results.** Every number in `report/` must be
   regenerable from a run directory by a script in `analysis/`.
2. **Seed splits are fixed:**
   | Split | Seeds | Use |
   |---|---|---|
   | dev | 0–19 | debugging, smoke tests |
   | val | 1000–1049 | **all** tuning (thresholds, weights, δ, K, H) |
   | test | 10000–10099 (extend to 10000–10299) | final numbers only |
   Nobody looks at test-split results until the configuration is frozen in a
   tagged commit. Tuning on test seeds invalidates the experiment.
3. **No privileged-information leakage.** Deployable policies and the
   difficulty detector receive only the agent's partial observation and its
   own history. Simulator state is available only to sandbox rollouts and the
   oracle. Code that uses it must be labeled `simulator-assisted` or `oracle`
   in its class name and in the results tables.
4. **Same base model for every compared method** within an experiment.
5. **Keep latency separate from compute.** Report wall-clock latency, model
   inference work, simulator CPU work, and GPU-seconds as separate quantities.
   Parallel branches can lower latency without lowering compute.
6. **Report negative results.** If speculation doesn't help, that is a finding.
   Investigate why. Don't tweak until it looks good.
7. **Statistics.** Use paired comparisons on common seeds, and 95% bootstrap
   confidence intervals. Don't claim significance without having computed it.

---

## 6. Experiments, results, and the cluster

- Every run writes `results/<YYYYMMDD-HHMMSS>-<config-name>-<gitsha>/`
  containing `manifest.json` (config, seeds, model name and digest, Ollama
  settings, package versions, git commit, hardware, timestamp) and
  `episodes.jsonl`.
- `results/` is **git-ignored**. Share finished runs by uploading them as
  release assets or to the shared drive, and link them in the issue.
  Summary CSVs that are small (under 1 MB) and feed the report may be
  committed under `report/data/`.
- Run experiments only from a **clean, committed** tree. The manifest records
  the git SHA, and a dirty tree makes the result unreproducible. The runner
  refuses to start on a dirty tree unless you pass `--allow-dirty`, which marks
  the run as non-final.
- **Laptops** are for editing, unit tests, and checks that take seconds. No
  long benchmarks or LLM sweeps on laptops.
- **GT Futurama cluster:**
  - All compute goes through `sbatch`. Never run compute on login nodes.
  - Use the `general` partition for this project, not `techfee`.
  - Before you submit a multi-GPU job, a job array, or anything with a
    walltime over 4 hours, post the partition, GPU count, and walltime in the
    issue and get an OK.
  - Ollama runs *inside* the job, bound to `127.0.0.1`. Model weights and
    caches go in `/dev/shm/$SLURM_JOB_ID` or `$TMPDIR`, never in `$HOME`, which
    has a 15 GB quota.
  - Keep secrets, tokens, and personal data off the cluster.
- **No paid cloud inference APIs.** All LLM calls go to Ollama.

---

## 7. Communication

- **The issue is the source of truth.** Decisions, blockers, and result links
  go in issue comments, not only in chat.
- A short status comment on your issue whenever you push, or every few hours,
  helps the others plan.
- Blocked on someone else's interface? Comment on their issue and label yours
  `status:blocked`.
- The tracking issue (labeled `tracking`) lists all work packages and their
  order. Pick unclaimed packages from it.

## Labels

| Label | Meaning |
|---|---|
| `status:in-progress` | Claimed and being worked on |
| `status:blocked` | Waiting on another issue |
| `status:needs-review` | PR open |
| `area:<name>` | One per row of the ownership map |
| `area:interfaces` | Touches the frozen contract |
| `tracking` | Meta / roadmap issue |
| `experiment` | Produces results; must link a run directory |
