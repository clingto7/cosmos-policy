"""Request complete checkpoints at optimizer boundaries without changing the recipe.

Run with ``python -m cosmos_policy.experiments.robot.bench2dex.checkpoint_control
RUN_DIR save`` (or ``save-and-stop``). A submitted request is not a saved checkpoint;
its receipt is written only after every rank finishes the synchronous save.
"""

import argparse
import json
import os
import uuid
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def submit(run_dir, action):
    if action not in ("save", "save-and-stop"):
        raise ValueError("expected save or save-and-stop")
    root = Path(run_dir) / "checkpoint-control"
    session = json.loads((root / "session.json").read_text())
    if session["state"] != "running":
        raise ValueError("training session is not accepting requests")
    request_id = uuid.uuid4().hex
    directory = root / session["id"]
    atomic_json(directory / f"{request_id}.request.json", {"action": action})
    return directory / f"{request_id}.result.json"


class CheckpointControl:
    """Rank zero owns the mailbox; all ranks receive the same boundary decision."""

    def __init__(self, run_dir):
        import torch.distributed as dist

        self.dist = dist
        self.distributed = dist.is_initialized()
        self.rank0 = not self.distributed or dist.get_rank() == 0
        self.root = Path(run_dir) / "checkpoint-control"
        self.session = None
        if self.rank0:
            self.session = uuid.uuid4().hex
            self.directory = self.root / self.session
            self.directory.mkdir(parents=True)
            self._state("running")

    def _state(self, state):
        atomic_json(self.root / "session.json", {"id": self.session, "state": state})

    def after_update(self, iteration, save, *, periodic=False):
        """Save once for overlapping periodic/manual requests; return whether to stop.

        Must be called by every rank, only after a complete optimizer update.
        ``save`` must be synchronous and run checkpoint validation callbacks.
        """
        command = [None]
        if self.rank0:
            for path in sorted(self.directory.glob("*.request.json")):
                request_id = path.name.removesuffix(".request.json")
                receipt = self.directory / f"{request_id}.result.json"
                if receipt.exists():
                    continue
                try:
                    action = json.loads(path.read_text())["action"]
                    if action not in ("save", "save-and-stop"):
                        raise ValueError("unknown checkpoint action")
                except (ValueError, KeyError, TypeError) as error:
                    atomic_json(receipt, {"status": "rejected", "error": str(error)})
                    continue
                command[0] = {"id": request_id, "action": action}
                break
        if self.distributed:
            self.dist.broadcast_object_list(command, src=0)
        request = command[0]
        if periodic or request:
            save()
            if self.distributed:
                self.dist.barrier()
        if request and self.rank0:
            atomic_json(
                self.directory / f"{request['id']}.result.json",
                {
                    "status": "saved",
                    "iteration": iteration,
                    "action": request["action"],
                    "checkpoint": f"checkpoints/iter_{iteration:09}",
                },
            )
            if request["action"] == "save-and-stop":
                self._state("stopping")
        return bool(request and request["action"] == "save-and-stop")

    def close(self):
        if self.rank0:
            self._state("stopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("action", choices=("save", "save-and-stop"))
    args = parser.parse_args()
    receipt = submit(args.run_dir, args.action)
    print(f"Request queued. Check receipt for successful save: {receipt}")


if __name__ == "__main__":
    main()
