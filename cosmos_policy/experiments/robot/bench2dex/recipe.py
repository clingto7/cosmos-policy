"""CPU-inspectable embodiment recipe layered on the existing Bench2Dex policy."""

import copy


def make_embodiment_experiment(base, manifest_path, robot_map_path, robot_key):
    experiment = copy.deepcopy(base)
    experiment["dataloader_train"]["dataset"] = {
        "_target_": "cosmos_policy.datasets.bench2dex_embodiment.load_embodiment_dataset",
        "manifest_path": manifest_path,
        "robot_map_path": robot_map_path,
        "chunk_size": 16,
        "seed": 0,
        "total_samples": 50000 * 64,
    }
    # The training entry point instantiates EmbodimentSampler once on this dataset.
    experiment["dataloader_train"].update(sampler=None, batch_size=16)
    experiment["trainer"].update(max_iter=50000, grad_accum_iter=2, run_validation=False)
    experiment["trainer"]["callbacks"]["embodiment_checkpoint"] = {
        "_target_": "cosmos_policy.bench2dex_proof.EmbodimentCheckpointCallback",
        "manifest_path": manifest_path,
    }
    experiment["checkpoint"].update(save_iter=5000, dcp_async_mode_enabled=False)
    experiment["job"].update(
        group="bench2dex_embodiment",
        name=f"cosmos_predict2_2b_480p_bench2dex_{robot_key}",
    )
    return experiment
