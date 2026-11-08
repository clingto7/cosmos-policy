#!/usr/bin/env python3
"""Replay recorded physical joint targets through the unmodified evaluation RPC."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cosmos_policy.datasets.bench2dex_audit import load_profile
from cosmos_policy.experiments.robot.bench2dex.runtime import Bench2DexTCPServer


def names(ds):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in ds[()]]


def read_actions(path, profile):
    with h5py.File(path, 'r') as f:
        steps = f['time/sim_step'][()]
        valid = f['action/action_valid'][()].astype(bool)
        valid &= f['frame_valid'][()].astype(bool)
        valid &= steps < int(f['meta/homing_start_sim_step'][()])
        ids = np.flatnonzero(valid)
        assert len(ids) and np.all(np.diff(ids) == 1), 'noncontiguous expert actions'
        assert np.all(np.diff(steps[ids]) == 3), 'not 20Hz actions'
        order = names(f['action/action_names'])
        assert len(set(order)) == len(order) and set(order) == set(profile.full_joint_names)
        actions = f['action/commanded'][()][ids][:, [order.index(n) for n in profile.full_joint_names]]
        assert np.isfinite(actions).all()
        qnames = names(f['robot/joint_names'])
        initial = f['robot/qpos'][0, :][[qnames.index(n) for n in profile.active_joint_names]]
        info = {'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'first_action_frame': int(ids[0]), 'last_action_frame': int(ids[-1]),
                'initial_to_first_action_physics_steps': int(steps[ids[0]] - steps[0]),
                'actions': len(actions), 'full_joint_names': list(profile.full_joint_names),
                'alignment': 'First valid commanded target at evaluation control tick zero; initial observation-only frame omitted; final pre-homing target held.'}
    return actions.astype(np.float32), initial, info


class Expert:
    def __init__(self, args, profile, anchor):
        self.args, self.profile, self.anchor = args, profile, anchor
        self.log = (args.output / 'rpc.jsonl').open('w', buffering=1)

    def emit(self, value):
        self.log.write(json.dumps(value) + '\n')

    def reset(self, seed=None):
        index = int(seed) - 100000000 - self.args.task * 100000
        assert 0 <= index < self.args.episodes
        self.actions, self.initial, info = read_actions(self.anchor / f'episode_{index:06d}.hdf5', self.profile)
        self.cursor, self.seed = 0, seed
        self.emit({'event': 'reset', 'seed': seed, **info})

    def get_action_chunk(self, observation):
        qpos = np.asarray(observation['joint_action']['vector']).reshape(-1)
        assert qpos.shape == self.initial.shape
        ids = np.minimum(np.arange(self.cursor, self.cursor + 16), len(self.actions) - 1)
        self.emit({'event': 'query', 'seed': self.seed, 'cursor': self.cursor,
                   'qpos': qpos.tolist(), 'tail_hold': int(np.sum(np.arange(self.cursor, self.cursor + 16) >= len(self.actions))),
                   'initial_qpos_max_abs_rad': float(np.max(np.abs(qpos - self.initial))) if self.cursor == 0 else None})
        self.cursor += 16
        return self.actions[ids]

    def update_after_action(self, observation):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-command', type=Path, required=True)
    p.add_argument('--robot-map', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', type=int, required=True, choices=[3, 79])
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--render-gpu', type=int, required=True)
    p.add_argument('--episodes', type=int, default=3, choices=range(1, 26))
    args = p.parse_args()
    assert str(args.output.resolve()).startswith('/inspire/hdd2/')
    args.output.mkdir(parents=True, exist_ok=False)
    command = json.loads(args.baseline_command.read_text())
    def value(flag):
        return command[command.index(flag) + 1]
    assert value('--device') == 'cpu' and value('--generalization-profile') == 'none'
    assert value('--start-episode') == '1' and int(value('--task-seed-id')) == args.task
    anchor = Path(value('--anchor-dir'))
    profile = load_profile(args.robot_map, value('--robot-key'))
    for index in range(args.episodes):
        read_actions(anchor / f'episode_{index:06d}.hdf5', profile)
    for flag, val in {'--remote-port': str(args.port), '--num-episodes': str(args.episodes),
                      '--policy-display-name': 'expert-original-commanded',
                      '--record-dir': str(args.output / 'episodes'),
                      '--output-dir': str(args.output / 'results')}.items():
        command[command.index(flag) + 1] = val
    command += ['--kit_args', f'--/renderer/activeGpu={args.render_gpu} --/renderer/multiGpu/enabled=false']
    (args.output / 'command.json').write_text(json.dumps(command, indent=2))
    adapter = Expert(args, profile, anchor)
    with Bench2DexTCPServer(('127.0.0.1', args.port), adapter) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with (args.output / 'simulator.log').open('w') as log:
                result = subprocess.run(command, cwd=Path(value('--task')).parents[1], stdout=log, stderr=subprocess.STDOUT)
            (args.output / 'exit.json').write_text(json.dumps({'returncode': result.returncode}))
        finally:
            server.shutdown()
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
