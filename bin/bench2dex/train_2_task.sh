#!/usr/bin/env bash
# 2 task(s): 4 GPUs x microbatch 16 x accumulation 2 = global batch 128.
# 32,000 optimizer updates; 4,096,000 total draws; 2,048,000 expected draws/task.
# Peak LR 1e-4; 800-step warmup; linear to 3e-5 at 24k, then constant 6e-6 until 32k.
# Required paths: MANIFEST ROBOT_MAP BASE_MODEL TOKENIZER OUTPUT RUN_NAME.
# Default is a read-only plan. Pass --execute to train, --smoke --execute to verify.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${MANIFEST:?Set the complete 2-task embodiment bundle}"
: "${ROBOT_MAP:?Set active_dof_maps.yml}"
: "${BASE_MODEL:?Set original Video2World .pt weights}"
: "${TOKENIZER:?Set tokenizer.pth}"
: "${OUTPUT:?Set private HDD2 training output root}"
: "${RUN_NAME:?Set a unique run name}"
cd "$repo"
# Reuse the selected Cosmos environment; install nothing.
export CUDA_HOME="${CUDA_HOME:-$("${PYTHON:-python}" -c 'import sys; print(sys.prefix)')}"
nvidia_libs="$("${PYTHON:-python}" -c 'import pathlib,sysconfig; print(":".join(str(p) for p in pathlib.Path(sysconfig.get_paths()["purelib"]).glob("nvidia/*/lib")))')"
export LD_LIBRARY_PATH="$nvidia_libs:${LD_LIBRARY_PATH:-}"
# Select the required physical GPUs externally via CUDA_VISIBLE_DEVICES.
exec "${PYTHON:-python}" -m cosmos_policy.experiments.robot.bench2dex.train_embodiment "$@" \
  --expected-tasks 2 --max-steps 32000 \
  --manifest "$MANIFEST" --robot-map "$ROBOT_MAP" \
  --base-model "$BASE_MODEL" --tokenizer "$TOKENIZER" \
  --output "$OUTPUT" --run-name "$RUN_NAME"
