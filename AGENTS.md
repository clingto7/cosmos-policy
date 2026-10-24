## Agent skills

### Issue tracker

Issues and specs live in this repository's GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the five default triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

Use a single root `CONTEXT.md` and `docs/adr/`, created when needed. See `docs/agents/domain.md`.

### Chuangzhi storage

Keep code on `/inspire/hdd/project/roboticsystem2/ky26063`. Put training datasets and checkpoints under the project's private `/inspire/hdd2/project/roboticsystem2/ky26063` tree. Do not write large training outputs to HDD; check the resolved filesystem of symlinks before choosing an output path.
