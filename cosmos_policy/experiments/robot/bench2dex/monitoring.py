"""Frozen dataset denominators for interpreting training exposure."""

from cosmos_policy.datasets.bench2dex_dataset import CAMERAS, CHUNK_SIZE, inspect_episode


def dataset_summary(bundle, profile, budget):
    tasks = {}
    for task in bundle['tasks']:
        episodes = []
        for entry in task['audit']['entries']:
            info = inspect_episode(entry['source'], profile)
            padded = sum(min(CHUNK_SIZE - 1, end - start) for start, end in info.action_runs)
            episodes.append({
                'source': str(info.path), 'sha256': entry['sha256'],
                'frames': info.frame_count, 'valid_windows': info.usable_frames,
                'excluded_tail_frames': info.excluded_frames,
                'invalid_task_frames': info.unusable_task_frames,
                'padded_action_windows': padded,
                'action_runs': list(info.action_runs),
            })
        windows = sum(e['valid_windows'] for e in episodes)
        if windows <= 0:
            raise ValueError('task has no valid training windows')
        tasks[task['task_id']] = {
            'episode_count': len(episodes), 'valid_windows': windows,
            'frames': sum(e['frames'] for e in episodes),
            'padded_action_windows': sum(e['padded_action_windows'] for e in episodes),
            'sampling_probability': 1 / budget.task_count,
            'expected_draws': budget.total_samples / budget.task_count,
            'expected_window_exposure': budget.total_samples / budget.task_count / windows,
            'episodes': episodes,
        }
    return {
        'schema': 'bench2dex-monitoring-v1', 'tasks': tasks,
        'action_horizon': CHUNK_SIZE, 'control_hz': 20, 'cameras': list(CAMERAS),
        'active_joint_names': list(profile.active_joint_names),
        'sampling': 'uniform task, then uniform valid start; with replacement',
        'exposure_definition': 'draws / valid start windows; not unique coverage or strict epochs',
        'action_metric_definition': 'single-step noisy denoising action latent L1/MSE; includes repeated padding; not physical joint error or rollout success',
        'smoothing': 'sample-weighted trailing 100 optimizer updates within current session; resets on resume',
        'validation': 'training metrics only; no held-out or closed-loop evaluation implied',
    }
