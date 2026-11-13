"""Offline audit failures and rollback-aware monitoring."""

import datetime as dt
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("run_status", Path(__file__).parents[1] / "bin/bench2dex/status.py")
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)


@pytest.fixture
def run(tmp_path, monkeypatch):
    identity = dict(source_sha256="abc", task_ids=["06", "12", "42"], world_size=8,
                    microbatch=12, grad_accum_iter=2, global_batch=192, total_samples=6144000,
                    max_iter=32000, save_iter=2500, extra_save_iters=[16500], peak_lr=1e-4,
                    scheduler={"warm_up_steps": [800, 0]})
    config = dict(trainer=dict(max_iter=32000, grad_accum_iter=2),
                  dataloader_train=dict(batch_size=12, dataset=dict(total_samples=6144000)),
                  checkpoint=dict(save_iter=2500, extra_save_iters=[16500]),
                  optimizer=dict(lr=1e-4), scheduler=identity["scheduler"])
    for name, value in [("run-identity", identity), ("resolved-training", config),
                        ("deployment", dict(commit="approved", source_sha256="abc", dirty=False))]:
        (tmp_path / (name + ".json")).write_text(json.dumps(value))
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    rows = [dict(event="update", iteration=17000, session="abandoned"),
            dict(event="start", iteration=16500, session="new", timestamp_utc=timestamp),
            dict(event="runtime", world_size=8, hostname="host-generation", session="new")]
    for step in [16501, 16502]:
        rows.append(dict(event="update", session="new", iteration=step, timestamp_utc=timestamp,
                         samples_seen=step * 192, task_samples_this_update={"06": 64, "12": 64, "42": 64},
                         loss=1.0, grad_norm_before_clip=2.0, lr_used=[1e-4], update_wall_seconds=3.4))
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(status, "held_lock", lambda path: True)
    return tmp_path


def test_resume_ignores_abandoned_updates_and_partial_append(run):
    with (run / "metrics.jsonl").open("a") as stream:
        stream.write('{"event":')
    result = status.inspect(run)
    assert result["ready"] and result["iteration"] == 16502
    assert result["next_save"] == 17500
    assert result["mean_seconds"] == 3.4 and result["speed_updates"] == 1


def test_wrong_resolved_schedule_rejected(run):
    path = run / "resolved-training.json"
    value = json.loads(path.read_text())
    value["checkpoint"]["extra_save_iters"] = []
    path.write_text(json.dumps(value))
    result = status.inspect(run)
    assert not result["ready"]
    assert "resolved extra saves differs from identity" in result["errors"]


def test_stale_and_unlocked_are_not_ready(run, monkeypatch):
    assert not status.inspect(run, max_age=-1)["ready"]
    monkeypatch.setattr(status, "held_lock", lambda path: False)
    assert not status.inspect(run)["ready"]


def test_nonfinite_and_wrong_samples_rejected(run):
    path = run / "metrics.jsonl"
    rows = status.journal(path)
    rows[-1].update(loss=float("nan"), samples_seen=1)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = status.inspect(run)
    assert not result["ready"] and "nonfinite metrics" in result["errors"]


def test_corrupt_committed_row_is_not_ignored(run):
    with (run / "metrics.jsonl").open("a") as stream:
        stream.write('{broken}\n')
    with pytest.raises(ValueError):
        status.inspect(run)


def test_marker_only_is_explicit_and_corrupt_marker_is_reported(run):
    directory = run / "checkpoints/iter_000016500"
    directory.mkdir(parents=True)
    marker = directory / "bench2dex-checkpoint.json"
    marker.write_text(json.dumps({"iteration": 16500}))
    result = status.inspect(run)
    assert result["checkpoint_check"] == "marker_only" and result["checkpoints"] == [16500]
    marker.write_text(json.dumps({"iteration": 1}))
    result = status.inspect(run)
    assert result["checkpoints"] == [] and len(result["invalid_checkpoints"]) == 1


def test_lock_probe_releases_its_lock(tmp_path):
    path = tmp_path / "launch.lock"
    path.touch()
    assert status.held_lock(path) is False
    with path.open("r") as owner:
        status.fcntl.flock(owner, status.fcntl.LOCK_EX | status.fcntl.LOCK_NB)
        assert status.held_lock(path) is True
    assert status.held_lock(path) is False
