"""CPU integration: audited multi-task data, sharded resume and physical actions."""

import copy
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from test_bench2dex_contract import PROFILE, write_episode
from torch.utils.data import DataLoader

from cosmos_policy.datasets.bench2dex_audit import audit_task
from cosmos_policy.datasets.bench2dex_embodiment import (
    EmbodimentDataset,
    EmbodimentSampler,
    load_embodiment_dataset,
    prepare_bundles,
    read_bundle,
    select_task,
)
from cosmos_policy.experiments.robot.bench2dex.checkpoints import file_hash
from cosmos_policy.experiments.robot.bench2dex.runtime import Bench2DexPolicyAdapter, _load_inputs


@pytest.fixture
def prepared(tmp_path):
    robot_map = tmp_path / "robots.yml"
    robot_map.write_text(
        "robots:\n  test_robot:\n    full_joint_names: [mimic, joint_b, joint_a]\n"
        "    active_joint_names: [joint_a, joint_b]\n"
    )
    catalog = []
    for task_id, homing, shift in (("03", 18, 0), ("79", 10, 200)):
        directory = tmp_path / f"{task_id}_task" / "replay-generalization"
        directory.mkdir(parents=True)
        template = tmp_path / f"{task_id}.hdf5"
        write_episode(template, homing=homing)
        with h5py.File(template, "a") as file:
            file["action/commanded"][:, :2] += shift
            file["robot/qpos"][:, [0, 2]] += shift
        for index in range(100):
            (directory / f"episode_{index:06}.hdf5").hardlink_to(template)
        description = f"Standard task {task_id} description."
        audit, _ = audit_task(directory, directory, PROFILE, description)
        manifest = tmp_path / f"task{task_id}.json"
        manifest.write_text(json.dumps(audit))
        embedding = tmp_path / f"task{task_id}.npz"
        np.savez(embedding, prompt=description, embedding=np.full((4, 8), int(task_id), dtype=np.float32))
        catalog.append({"manifest": manifest.name, "embedding": embedding.name})
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog))
    grouping = tmp_path / "taskstats.json"
    grouping.write_text(
        json.dumps({"per_task": [{"task": f"{i}_task", "robot_key": "test_robot"} for i in ("03", "79")]})
    )
    (bundle_path,) = prepare_bundles(catalog_path, str(robot_map), tmp_path / "bundles", grouping)
    return SimpleNamespace(
        root=tmp_path, bundle=bundle_path, robot_map=robot_map, catalog=catalog_path, grouping=grouping
    )


def test_audit_union_prompt_and_physical_runtime_roundtrip(prepared):
    bundle, _, statistics = read_bundle(prepared.bundle, str(prepared.robot_map))
    np.testing.assert_array_equal(statistics.actions_min, [1, 101])
    np.testing.assert_array_equal(statistics.actions_max, [209, 309])
    # Invalid frame zero and homing frames do not contribute extrema.
    dataset = load_embodiment_dataset(str(prepared.bundle), str(prepared.robot_map))
    assert [len(task) for task in dataset.datasets] == [1700, 900]
    for task in dataset.datasets:
        task.image_size = 4
    samples = [task[0] for task in dataset.datasets]
    for task, sample in zip(bundle["tasks"], samples):
        assert sample["command"] == task["audit"]["task_description"]
        assert np.all(sample["t5_text_embeddings"] == int(task["task_id"]))
        payload = {
            "language": sample["command"],
            "joint_action": {"vector": statistics.denormalize_proprio(sample["proprio"])},
            "observation": {camera: {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)} for camera in sample["camera_order"]},
        }
        adapter = Bench2DexPolicyAdapter(
            lambda obs, prompt, seed: sample["actions"],
            PROFILE,
            statistics,
            sample["command"],
            runtime_joint_names=PROFILE.full_joint_names,
        )
        output = adapter.get_action_chunk(payload)
        physical = statistics.denormalize_actions(sample["actions"])
        np.testing.assert_allclose(output[:, 1], physical[:, 1])
        np.testing.assert_allclose(output[:, 2], physical[:, 0])
        np.testing.assert_allclose(output[:, 0], 0.25 - physical[:, 1])
    batch = next(iter(DataLoader(dataset, batch_size=4, num_workers=0)))
    assert batch["actions"].shape == (4, 16, 2)
    assert batch["t5_text_embeddings"].shape == (4, 4, 8)
    assert len(set(batch["__key__"])) >= 1


def test_uniform_tasks_despite_unequal_lengths_and_reproducible_resume():
    dataset = EmbodimentDataset([range(1), range(100), range(1000)], ["03", "06", "07"], total_samples=19200)
    draws = np.array([dataset.locate(i) for i in range(len(dataset))])
    counts = np.bincount(draws[:, 0], minlength=3)
    assert np.all(np.abs(counts - 6400) < 250)
    assert len(set(draws[draws[:, 0] == 2, 1])) > 990
    ranks = [EmbodimentSampler(dataset, 2, rank) for rank in (0, 1)]
    before = [list(sampler) for sampler in ranks]
    assert set(before[0]).isdisjoint(before[1])
    assert sorted(before[0] + before[1]) == list(range(len(dataset)))
    for rank, sampler in enumerate(ranks):
        sampler.set_start_iteration(17)
        after = list(sampler)
        assert after == before[rank][17 * 32 :]
        assert [dataset.locate(i) for i in after[:20]] == [dataset.locate(i) for i in before[rank][544:564]]
        sampler.set_start_iteration(300)
        assert len(sampler) == 0 and list(sampler) == []
    with pytest.raises(ValueError, match="2/4/8 data-parallel"):
        EmbodimentSampler(dataset, 1, 0)


@pytest.mark.parametrize("mutation", ["robot", "joints", "duplicate", "stats", "prompt", "camera"])
def test_bundle_rejects_contract_mismatches(prepared, mutation):
    bundle = json.loads(prepared.bundle.read_text())
    audit = bundle["tasks"][1]["audit"]
    if mutation == "robot":
        bundle["robot_key"] = "other"
    elif mutation == "joints":
        audit["active_joint_names"].reverse()
    elif mutation == "duplicate":
        audit["entries"][0] = copy.deepcopy(bundle["tasks"][0]["audit"]["entries"][0])
    elif mutation == "stats":
        bundle["statistics"]["actions_max"][0] += 1
    elif mutation == "prompt":
        audit["task_description"] = bundle["tasks"][0]["audit"]["task_description"]
    else:
        audit["camera_names"].reverse()
    prepared.bundle.write_text(json.dumps(bundle))
    with pytest.raises(ValueError):
        read_bundle(prepared.bundle, str(prepared.robot_map))


def test_prepare_rejects_missing_group_data_drift_and_rewrite(prepared):
    with pytest.raises(FileExistsError):
        prepare_bundles(prepared.catalog, str(prepared.robot_map), prepared.bundle.parent, prepared.grouping)
    catalog = json.loads(prepared.catalog.read_text())
    prepared.catalog.write_text(json.dumps(catalog[:1]))
    with pytest.raises(ValueError, match="complete task group"):
        prepare_bundles(prepared.catalog, str(prepared.robot_map), prepared.root / "missing", prepared.grouping)
    prepared.catalog.write_text(json.dumps(catalog))
    with h5py.File(prepared.root / "03.hdf5", "a") as file:
        file["action/commanded"][1, 0] = -999
    with pytest.raises(ValueError, match="changed since audit"):
        prepare_bundles(prepared.catalog, str(prepared.robot_map), prepared.root / "drift", prepared.grouping)


def test_runtime_loads_shared_stats_and_rejects_task_or_embedding_mismatch(prepared, monkeypatch):
    import cosmos_policy.experiments.robot.bench2dex.checkpoints as checkpoints

    # DCP storage validation has independent serialization tests. Isolate this data-loading seam.
    monkeypatch.setattr(checkpoints, "require_checkpoint_contract", lambda *a, **kw: None)
    args = SimpleNamespace(
        manifest=str(prepared.root / "task79.json"),
        embedding=str(prepared.root / "task79.npz"),
        robot_map=str(prepared.robot_map),
        task_id="79",
        checkpoint=str(prepared.root / "checkpoint"),
        embodiment_manifest=str(prepared.bundle),
    )
    profile, stats, prompt, embedding, task_id = _load_inputs(args)
    assert task_id == "79" and profile.robot_key == "test_robot"
    assert stats.actions_min[0] == 1  # task79 alone has a minimum of 201
    assert np.all(embedding == 79) and "79" in prompt
    np.savez(args.embedding, prompt=prompt, embedding=np.zeros((4, 8)))
    with pytest.raises(ValueError, match="differs from the training bundle"):
        _load_inputs(args)
    bundle = json.loads(prepared.bundle.read_text())
    with pytest.raises(ValueError, match="does not belong"):
        select_task(bundle, "06")
    assert file_hash(args.embedding) != bundle["tasks"][1]["embedding_sha256"]


def test_recipe_and_launcher_freeze_budget_without_mutating_per_task(prepared, monkeypatch):
    from pathlib import Path

    from cosmos_policy._src.imaginaire.lazy_config import LazyDict, instantiate
    from cosmos_policy.experiments.robot.bench2dex.recipe import make_embodiment_experiment
    from cosmos_policy.experiments.robot.bench2dex.train_embodiment import plan_run

    original = {
        "dataloader_train": {"batch_size": 1, "dataset": "per-task", "sampler": "ordinary"},
        "trainer": {"callbacks": {"finite": {"_target_": "finite"}}},
        "checkpoint": {"load_training_state": False},
        "job": {},
        "scheduler": {"cycle_lengths": [30000, 100000000000000]},
    }
    original = LazyDict(original)
    recipe = make_embodiment_experiment(original, str(prepared.bundle), str(prepared.robot_map), "test_robot")
    dataset = instantiate(recipe["dataloader_train"]["dataset"])
    assert isinstance(dataset, EmbodimentDataset) and len(dataset.datasets) == 2
    assert original["dataloader_train"]["dataset"] == "per-task"
    assert recipe["dataloader_train"]["batch_size"] == 16
    assert recipe["trainer"]["grad_accum_iter"] == 2
    assert recipe["trainer"]["max_iter"] == 32000
    assert recipe["trainer"]["run_validation"] is False
    for name in ("every_n_sample_reg", "every_n_sample_ema"):
        assert recipe["trainer"]["callbacks"][name]["every_n"] == 32001
        assert recipe["trainer"]["callbacks"][name]["run_at_start"] is False
    assert recipe["checkpoint"]["save_iter"] == 5000
    assert recipe["checkpoint"]["dcp_async_mode_enabled"] is False
    assert recipe["scheduler"]["cycle_lengths"] == [24000, 100000000000000]
    assert recipe["scheduler"]["_target_"] == "cosmos_policy._src.imaginaire.functional.lr_scheduler.LambdaLinearScheduler"
    assert recipe["scheduler"]["f_start"] == [1e-6, 0.06]
    assert recipe["scheduler"]["f_max"] == [1.0, 0.06]
    assert recipe["scheduler"]["f_min"] == [0.3, 0.06]
    assert original["scheduler"]["cycle_lengths"][0] == 30000
    assert len(dataset) == 128 * 32000
    weights, tokenizer = prepared.root / "base.pt", prepared.root / "tokenizer.pth"
    weights.write_bytes(b"test-weight-identity")
    tokenizer.write_bytes(b"test-tokenizer-identity")
    args = SimpleNamespace(
        manifest=prepared.bundle,
        robot_map=prepared.robot_map,
        base_model=weights,
        tokenizer=tokenizer,
        output=prepared.root / "output",
        run_name="test-run",
        repo=Path(__file__).resolve().parents[1],
        smoke=False,
    )
    command, identity, run_dir = plan_run(args)
    assert identity["global_batch"] == 128 and identity["max_iter"] == 32000
    assert "--nproc_per_node=4" in command and "checkpoint.save_iter=5000" in command
    assert not args.output.exists() and run_dir.name == "test-run"
    args.max_steps = 12000
    _, shorter, _ = plan_run(args)
    assert shorter["total_samples"] == 128 * 12000
    assert shorter["scheduler"]["cycle_lengths"] == [9000, 100000000000000]
    monkeypatch.setenv("BENCH2DEX_MAX_STEPS", "12000")
    shorter_recipe = make_embodiment_experiment(original, str(prepared.bundle), str(prepared.robot_map), "test_robot")
    assert shorter_recipe["dataloader_train"]["dataset"]["total_samples"] == 128 * 12000
    assert shorter_recipe["scheduler"]["warm_up_steps"] == [300, 0]
    assert shorter_recipe["trainer"]["max_iter"] == 12000
    args.expected_tasks = 3
    with pytest.raises(ValueError, match="task count"):
        plan_run(args)
    args.expected_tasks = 2
    args.smoke = True
    _, smoke_identity, _ = plan_run(args)
    assert smoke_identity["max_iter"] == 4 and smoke_identity != identity
    args.smoke_steps = 8
    _, stage_smoke, _ = plan_run(args)
    assert stage_smoke["max_iter"] == 8 and stage_smoke["total_samples"] == 1024
    assert stage_smoke["first_cycle_steps"] == 6 and stage_smoke["warmup_steps"] == 1
    weights.write_bytes(b"changed")
    _, changed, _ = plan_run(args)
    assert changed["base_model_sha256"] != identity["base_model_sha256"]


def test_single_task_bundle_and_embedding_drift(prepared):
    catalog = json.loads(prepared.catalog.read_text())[:1]
    prepared.catalog.write_text(json.dumps(catalog))
    prepared.grouping.write_text(json.dumps({"per_task": [{"task": "03_task", "robot_key": "test_robot"}]}))
    (path,) = prepare_bundles(prepared.catalog, str(prepared.robot_map), prepared.root / "single", prepared.grouping)
    dataset = load_embodiment_dataset(str(path), str(prepared.robot_map))
    assert len(dataset.datasets) == 1 and all(dataset.locate(i)[0] == 0 for i in range(100))
    bundle = json.loads(path.read_text())
    assert bundle["statistics"] == bundle["tasks"][0]["audit"]["statistics"]
    (prepared.root / "task03.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="embedding changed"):
        load_embodiment_dataset(str(path), str(prepared.robot_map))


def test_worker_prefetch_does_not_change_draws(prepared):
    dataset = load_embodiment_dataset(str(prepared.bundle), str(prepared.robot_map))
    for task in dataset.datasets:
        task.image_size = 4
    baseline = next(iter(DataLoader(dataset, batch_size=8, num_workers=0)))
    iterator = iter(DataLoader(dataset, batch_size=8, num_workers=2, multiprocessing_context="spawn"))
    try:
        parallel = next(iterator)
    finally:
        del iterator
    assert parallel["task_id"] == baseline["task_id"]
    assert parallel["__key__"] == baseline["__key__"]
    np.testing.assert_array_equal(parallel["actions"].numpy(), baseline["actions"].numpy())


def test_launcher_idempotence_restore_probe_and_run_identity(prepared, monkeypatch):
    from pathlib import Path

    from test_bench2dex_checkpoint_retention import save_checkpoint

    from cosmos_policy.experiments.robot.bench2dex import train_embodiment as launcher
    from cosmos_policy.experiments.robot.bench2dex.checkpoints import mark_and_prune

    weights, tokenizer = prepared.root / "base.pt", prepared.root / "tokenizer.pth"
    weights.write_bytes(b"test-base")
    tokenizer.write_bytes(b"test-tokenizer")
    args = SimpleNamespace(
        manifest=prepared.bundle,
        robot_map=prepared.robot_map,
        base_model=weights,
        tokenizer=tokenizer,
        output=prepared.root / "output",
        run_name="smoke",
        repo=Path(__file__).resolve().parents[1],
        smoke=True,
        verify_restore=False,
    )
    command, identity, run_dir = launcher.plan_run(args)
    monkeypatch.setattr(launcher, "check_training_devices", lambda smoke, world_size: None)
    launches = []

    def simulate_worker(command, **kwargs):
        # Replace only the GPU subprocess; admission and DCP validation remain real.
        launches.append(command)
        assert kwargs["env"]["BENCH2DEX_EMBODIMENT_MANIFEST"] == str(prepared.bundle)
        checkpoint = save_checkpoint(run_dir / "checkpoints", 4)
        mark_and_prune(checkpoint.parent, 4, prepared.bundle)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", simulate_worker)
    launcher.execute_run(args, command, identity, run_dir)
    launcher.execute_run(args, command, identity, run_dir)
    assert len(launches) == 1  # completed run is not accidentally retrained
    args.verify_restore = True
    launcher.execute_run(args, command, identity, run_dir)
    assert len(launches) == 2
    with pytest.raises(ValueError, match="different data, weights, code"):
        launcher.execute_run(args, command, {**identity, "base_model_sha256": "different"}, run_dir)


def test_bundle_rechecks_complete_group_and_reference(prepared):
    bundle = json.loads(prepared.bundle.read_text())
    bundle['tasks'] = bundle['tasks'][:1]
    bundle['statistics'] = bundle['tasks'][0]['audit']['statistics']
    prepared.bundle.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match='complete task group'):
        read_bundle(prepared.bundle, str(prepared.robot_map))
    prepared.grouping.write_text('{}')
    with pytest.raises(ValueError, match='grouping reference'):
        read_bundle(prepared.bundle, str(prepared.robot_map))
