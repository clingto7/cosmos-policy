"""Append-only optimizer-update metrics with checkpoint-aligned resume accounting."""

import json
import math
import os
import uuid
from pathlib import Path


class MetricsJournal:
    def __init__(self, path, task_ids, global_batch, resume_iteration):
        self.path = Path(path)
        self.task_ids = list(task_ids)
        self.global_batch = global_batch
        self.session = uuid.uuid4().hex
        self.counts = dict.fromkeys(self.task_ids, 0)
        self.iteration = resume_iteration
        if resume_iteration:
            found = None
            with self.path.open() as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # A killed writer may leave a partial line.
                    if row.get("event") == "update" and row.get("iteration") == resume_iteration:
                        found = row
            if found is None or found["samples_seen"] != resume_iteration * global_batch:
                raise ValueError("resume checkpoint has no matching metrics accounting")
            self.counts = found["task_samples_seen"]
            if set(self.counts) != set(task_ids) or sum(self.counts.values()) != resume_iteration * global_batch:
                raise ValueError("resume task sample accounting differs from checkpoint")
        # Separate any partial last line before appending a new session.
        with self.path.open("a") as stream:
            stream.write("\n")
        self.append({"event": "start", "iteration": resume_iteration, "task_samples_seen": self.counts})

    def append(self, record):
        with self.path.open("a") as stream:
            stream.write(json.dumps({"session": self.session, **record}, allow_nan=False) + "\n")

    def update(self, iteration, loss, lr_used, lr_next, task_counts, diagnostics=None):
        if iteration != self.iteration + 1 or set(task_counts) != set(self.task_ids):
            raise ValueError("nonsequential optimizer metrics or changed task IDs")
        if sum(task_counts.values()) != self.global_batch or any(v < 0 for v in task_counts.values()):
            raise ValueError("observed samples do not match global batch")
        if not all(math.isfinite(v) for v in [loss, *lr_used, *lr_next]):
            raise ValueError("nonfinite training metrics")
        diagnostics = diagnostics or {}
        # Validate before advancing in-memory state or writing a partial JSON record.
        json.dumps(diagnostics, allow_nan=False)
        self.counts = {k: self.counts[k] + task_counts[k] for k in self.task_ids}
        self.iteration = iteration
        self.append(
            {
                "event": "update",
                "iteration": iteration,
                "samples_seen": iteration * self.global_batch,
                "task_samples_seen": self.counts,
                "task_samples_this_update": task_counts,
                "loss": loss,
                "lr_used": lr_used,
                "lr_next": lr_next,
                **diagnostics,
            }
        )

    def sync(self):
        with self.path.open("a") as stream:
            stream.flush()
            os.fsync(stream.fileno())
