#!/usr/bin/env python3
"""Run the CPU contracts and reject new core dumps even when pytest passes."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = [
    "tests/test_bench2dex_contract.py",
    "tests/test_bench2dex_budget.py",
    "tests/test_bench2dex_embodiment.py",
    "tests/test_bench2dex_checkpoint_retention.py",
    "tests/test_bench2dex_save_schedule.py",
    "tests/test_bench2dex_status.py",
]


def cores():
    found = {}
    for parent, directories, files in os.walk(ROOT):
        directories[:] = [d for d in directories if d not in {".git", ".venv", "node_modules"}]
        for name in files:
            if name == "core" or (name.startswith("core.") and name[5:].isdigit()):
                path = Path(parent) / name
                stat = path.stat()
                found[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return found


def main():
    before = cores()
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", *TESTS, "-p", "no:cacheprovider"], cwd=ROOT)
    after = cores()
    changed = [path for path in after if after[path] != before.get(path)]
    if changed:
        print("New/changed core dumps (preserved):", *changed, sep="\n", file=sys.stderr)
    raise SystemExit(result.returncode or bool(changed))


if __name__ == "__main__":
    main()
