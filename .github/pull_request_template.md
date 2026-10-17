Closes #

## What changed

## Files touched
<!-- Must match the issue's "Files touched" list. If it grew, update the issue first. -->

## How I verified it
<!-- Commands run and their results. For experiment PRs, give the run directory and manifest git SHA. -->

## Checklist
- [ ] Rebased on latest `main`
- [ ] `make check` passes locally (no Ollama required)
- [ ] Diff touches only the files declared in the issue
- [ ] No privileged simulator state reaches deployable policies or the detector (or the class is labeled `oracle` / `simulator-assisted`)
- [ ] No tuning on test seeds; no hand-edited results
