#!/usr/bin/env python3
"""Aggregate completed four-channel Cosmos evaluations into leaderboard metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CHANNELS = ("none", "cov_only", "inv_only", "inv_cov")
# Bench2Dex/scenes/task.md release catalog. Scene 86 is not in the 26-task release.
REFERENCE_TASK_IDS = frozenset("03 06 07 08 09 12 21 22 24 26 27 32 34 42 43 44 51 60 61 62 64 67 73 76 79 80".split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", required=True, type=Path,
                        help="Directory containing one completed task evaluation directory per task")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-all-26", action="store_true")
    args = parser.parse_args()

    tasks = {}
    for path in sorted(args.eval_root.glob("*/summary.json")):
        result = json.loads(path.read_text())
        task_id = str(result["task_id"])
        if task_id in tasks:
            parser.error(f"duplicate task {task_id}")
        channels = result["channels"]
        if result.get("seed_namespace") != "final":
            parser.error(f"task {task_id} uses pilot seeds and cannot enter the final aggregate")
        if set(channels) != set(CHANNELS) or any(channels[ch]["episodes"] != 50 for ch in CHANNELS):
            parser.error(f"task {task_id} is not a completed 50×4 evaluation")
        tasks[task_id] = result
    if not tasks:
        parser.error("no completed task evaluations found")
    unknown = set(tasks) - REFERENCE_TASK_IDS
    if unknown:
        parser.error(f"tasks outside the 26-task release catalog: {sorted(unknown)}")
    missing = REFERENCE_TASK_IDS - set(tasks)
    if args.require_all_26 and missing:
        parser.error(f"missing release tasks: {sorted(missing)}")

    task_rows = []
    for task_id, result in sorted(tasks.items()):
        row = {"task_id": task_id, "task_name": result["task_name"],
               "checkpoint_id": result["checkpoint_id"],
               "sr": result["per_task_sr"], "lscr": result["per_task_lscr"],
               "eff": result["per_task_eff"]}
        row.update({f"sr_{channel}": result["channels"][channel]["sr"] for channel in CHANNELS})
        task_rows.append(row)
    efficiencies = [row["eff"] for row in task_rows if row["eff"] is not None]
    aggregate = {
        "task_count": len(task_rows),
        "complete_leaderboard": not missing,
        "overall_sr": sum(row["sr"] for row in task_rows) / len(task_rows),
        "overall_lscr": sum(row["lscr"] for row in task_rows) / len(task_rows),
        "overall_eff": sum(efficiencies) / len(efficiencies) if efficiencies else None,
        "per_task": task_rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "overall.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    with (args.output / "per_task.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=task_rows[0].keys())
        writer.writeheader()
        writer.writerows(task_rows)
    print(f"tasks={len(task_rows)} overall_sr={aggregate['overall_sr']:.4f}")


if __name__ == "__main__":
    main()
