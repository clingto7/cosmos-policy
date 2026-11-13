#!/usr/bin/env python3
"""Read-only run audit. Uses stdlib; optional full checkpoint checks use Cosmos env."""

import argparse
import datetime as dt
import fcntl
import json
import math
import statistics
import sys
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def journal(path):
    """Ignore only a final incomplete append, never malformed committed rows."""
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(b"\n")]


def held_lock(path):
    if not path.exists():
        return False
    with path.open("r") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def inspect(run, max_age=300, verify_checkpoints=False):
    run = Path(run)
    result = {"run_dir": str(run), "observed_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    errors = []
    required = ["run-identity.json", "deployment.json", "resolved-training.json", "metrics.jsonl"]
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        return {**result, "ready": False, "errors": ["missing: " + ", ".join(missing)]}
    identity = read_json(run / required[0])
    deploy = read_json(run / required[1])
    config = read_json(run / required[2])

    def check(ok, message):
        if not ok:
            errors.append(message)

    check(deploy["source_sha256"] == identity["source_sha256"], "deployment source differs from identity")
    check(deploy.get("dirty") is False, "deployment was dirty or unverified")
    pairs = [
        (config["trainer"]["max_iter"], identity["max_iter"], "max_iter"),
        (config["trainer"]["grad_accum_iter"], identity["grad_accum_iter"], "accumulation"),
        (config["dataloader_train"]["batch_size"], identity["microbatch"], "microbatch"),
        (config["dataloader_train"]["dataset"]["total_samples"], identity["total_samples"], "stream length"),
        (config["checkpoint"]["save_iter"], identity["save_iter"], "save interval"),
        (config["checkpoint"].get("extra_save_iters", []), identity.get("extra_save_iters", []), "extra saves"),
        (config["optimizer"]["lr"], identity["peak_lr"], "peak LR"),
    ]
    for actual, expected, name in pairs:
        check(actual == expected, "resolved " + name + " differs from identity")
    for key, value in identity["scheduler"].items():
        check(config["scheduler"].get(key) == value, "scheduler differs: " + key)
    batch = identity["world_size"] * identity["microbatch"] * identity["grad_accum_iter"]
    check(batch == identity["global_batch"], "global batch arithmetic mismatch")
    check(batch * identity["max_iter"] == identity["total_samples"], "sample budget mismatch")
    rows = journal(run / "metrics.jsonl")
    starts = [i for i, row in enumerate(rows) if row.get("event") == "start"]
    if not starts:
        return {**result, "ready": False, "errors": errors + ["no start event"]}
    current = rows[starts[-1]:]
    start = current[0]
    current = [row for row in current if row.get("session") == start["session"]]
    updates = [row for row in current if row.get("event") == "update"]
    runtime = [row for row in current if row.get("event") == "runtime"]
    check(bool(runtime) and runtime[-1]["world_size"] == identity["world_size"], "runtime world size missing/differs")
    for offset, row in enumerate(updates, 1):
        step = start["iteration"] + offset
        check(row["iteration"] == step, "non-contiguous updates in latest session")
        check(all(math.isfinite(row[k]) for k in ["loss", "grad_norm_before_clip"]), "nonfinite metrics")
        rates = row.get("lr_used", [])
        check(bool(rates) and all(math.isfinite(rate) and rate >= 0 for rate in rates), "missing/nonfinite LR")
        check(row["samples_seen"] == row["iteration"] * batch, "committed sample count mismatch")
        check(set(row["task_samples_this_update"]) == set(identity["task_ids"]), "task membership mismatch")
        check(sum(row["task_samples_this_update"].values()) == batch, "per-update samples mismatch")
    latest = updates[-1] if updates else start
    timestamp = dt.datetime.fromisoformat(latest["timestamp_utc"])
    age = (dt.datetime.now(dt.timezone.utc) - timestamp).total_seconds()
    # The first update after a start includes initialization; retain later waits/saves.
    times = [row["update_wall_seconds"] for row in updates[1:][-100:]]
    checkpoints = []
    invalid = []
    if verify_checkpoints:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from cosmos_policy.experiments.robot.bench2dex.checkpoints import require_checkpoint_contract
    for marker in sorted((run / "checkpoints").glob("iter_*/bench2dex-checkpoint.json")):
        try:
            if marker.parent.is_symlink():
                raise ValueError("symlink checkpoint")
            step = int(marker.parent.name.removeprefix("iter_"))
            check_marker = read_json(marker)
            if check_marker["iteration"] != step:
                raise ValueError("marker iteration differs")
            if verify_checkpoints:
                manifest = Path(config["dataloader_train"]["dataset"]["manifest_path"])
                require_checkpoint_contract(marker.parent, manifest)
            checkpoints.append(step)
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            invalid.append({"path": str(marker.parent), "error": str(exc)})
    iteration = latest["iteration"]
    interval = identity["save_iter"]
    future = [((iteration // interval) + 1) * interval, identity["max_iter"]]
    future += [step for step in identity.get("extra_save_iters", []) if step > iteration]
    future = [step for step in future if iteration < step <= identity["max_iter"]]
    live = held_lock(run / "launch.lock")
    result.update(
        tasks=identity["task_ids"], commit=deploy["commit"], source_sha256=deploy["source_sha256"],
        hostname=runtime[-1].get("hostname") if runtime else None,
        session=start["session"], resumed_from=start["iteration"], iteration=iteration,
        target=identity["max_iter"], lr=latest.get("lr_used"), loss=latest.get("loss"),
        rolling_l1=latest.get("task_action_l1_rolling100"),
        age_seconds=round(age, 1), lock_held=live,
        mean_seconds=statistics.mean(times) if times else None,
        median_seconds=statistics.median(times) if times else None, speed_updates=len(times),
        checkpoints=checkpoints, checkpoint_check="full" if verify_checkpoints else "marker_only",
        invalid_checkpoints=invalid, next_save=min(future) if future else None,
        errors=errors, ready=not errors and len(updates) >= 2 and live and 0 <= age <= max_age,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--require-ready", action="store_true", help="Exit nonzero unless all runs pass startup audit")
    parser.add_argument("--verify-checkpoints", action="store_true", help="Validate full DCP artifacts with Cosmos environment")
    parser.add_argument("--max-age", type=int, default=300)
    parser.add_argument("--expected-commit", help="Approved deployment revision, full SHA")
    parser.add_argument("--expected-source", help="Approved production source SHA256")
    args = parser.parse_args()
    results = []
    for run in args.runs:
        try:
            row = inspect(run, args.max_age, args.verify_checkpoints)
            for key, expected in [("commit", args.expected_commit), ("source_sha256", args.expected_source)]:
                if expected is not None and row.get(key) != expected:
                    row["errors"].append("differs from approved " + key)
                    row["ready"] = False
            results.append(row)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            results.append({"run_dir": str(run), "ready": False, "errors": [str(exc)]})
    print(json.dumps(results, indent=2))
    if args.require_ready and not all(row["ready"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
