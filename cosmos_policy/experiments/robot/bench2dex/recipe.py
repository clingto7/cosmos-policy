"""CPU-inspectable embodiment recipe layered on the existing Bench2Dex policy."""

import copy
import json
import os
from pathlib import Path

from cosmos_policy.experiments.robot.bench2dex.budget import Budget


def make_embodiment_experiment(base, manifest_path, robot_map_path, robot_key):
    task_ids = [t["task_id"] for t in json.loads(Path(manifest_path).read_text())["tasks"]]
    budget = Budget(len(task_ids), int(os.environ.get("BENCH2DEX_MAX_STEPS", "32000")),
                    os.environ.get("BENCH2DEX_SINGLE_TASK_FOUR_GPU") == "1")
    experiment = copy.deepcopy(base)
    experiment["dataloader_train"]["dataset"] = {
        "_target_": "cosmos_policy.datasets.bench2dex_embodiment.load_embodiment_dataset",
        "manifest_path": manifest_path,
        "robot_map_path": robot_map_path,
        "chunk_size": 16,
        "seed": 0,
        "total_samples": budget.total_samples,
    }
    # The training entry point instantiates EmbodimentSampler once on this dataset.
    experiment["dataloader_train"].update(sampler=None, batch_size=budget.microbatch)
    experiment["trainer"].update(max_iter=budget.max_steps, grad_accum_iter=budget.grad_accum_iter, run_validation=False)
    # Hydra can merge sampling callbacks from the base experiment later.
    # Override both explicitly so neither fires within this training budget.
    for name in ("every_n_sample_reg", "every_n_sample_ema"):
        experiment["trainer"]["callbacks"].setdefault(name, {})
        experiment["trainer"]["callbacks"][name].update(
            _target_="cosmos_policy._src.predict2.callbacks.every_n_draw_sample.EveryNDrawSample",
            every_n=budget.max_steps + 1,
            step_size=1,
            run_at_start=False,
        )
    experiment["trainer"]["callbacks"]["embodiment_checkpoint"] = {
        "_target_": "cosmos_policy.bench2dex_proof.EmbodimentCheckpointCallback",
        "manifest_path": manifest_path,
    }
    experiment["trainer"]["callbacks"]["embodiment_metrics"] = {
        "_target_": "cosmos_policy.bench2dex_proof.EmbodimentMetricsCallback",
        "task_ids": task_ids,
    }
    experiment["optimizer"] = {**experiment.get("optimizer", {}), "lr": 1e-4}
    experiment["scheduler"] = {
        "_target_": "cosmos_policy._src.imaginaire.functional.lr_scheduler.LambdaLinearScheduler",
        **budget.scheduler(),
    }
    experiment["checkpoint"].update(save_iter=5000, dcp_async_mode_enabled=False)
    experiment["job"].update(
        group="bench2dex_embodiment",
        name=f"cosmos_predict2_2b_480p_bench2dex_{robot_key}",
    )
    return experiment
