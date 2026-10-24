#!/usr/bin/env bash
set -Eeuo pipefail

# Train one task 03 trajectory toward 15k, 20k, and 25k evaluation budgets.
# Save every 1k for rolling resume; prune after verifying a newer checkpoint.
# Re-running this launcher uses the same output root; Cosmos DCP resumes from
# its latest completed checkpoint when one exists.

readonly ROOT="/inspire/hdd/project/roboticsystem2/ky26063/bench2dex_stuff"
readonly DATA_ROOT="/inspire/hdd2/project/roboticsystem2/ky26063/bench2dex_stuff/hqb/task03"
readonly REPO="$ROOT/hqb/cosmos4dex"
readonly TASK03="$ROOT/hqb/task03"
readonly ENV_PREFIX="$ROOT/miniconda3/envs/cosmos_policy"
readonly RUN_NAME="task03_budget_effbatch2_25k_20261022_r2"
readonly OUTPUT_ROOT="$DATA_ROOT/output-budget-effbatch2-25k"
readonly LOG_FILE="$TASK03/train-budget-effbatch2-25k-r2.log"
readonly RUN_DIR="$OUTPUT_ROOT/cosmos_policy/bench2dex/$RUN_NAME"
readonly INITIAL_CHECKPOINT="$DATA_ROOT/output/cosmos_policy/bench2dex/cosmos_predict2_2b_480p_bench2dex_task03/checkpoints/iter_000000009"
readonly BASE_MODEL="$ROOT/policy_ckpt/pretrained/cosmos-predict2-2b-video2world"

source "$ROOT/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

export PATH="$ENV_PREFIX/bin:$PATH"
export COSMOS_INTERNAL=1
export CUDA_HOME="$ENV_PREFIX"
export COSMOS_POLICY_PRETRAINED_CHECKPOINT="$BASE_MODEL/model-480p-16fps.pt"
export COSMOS_POLICY_TOKENIZER_CHECKPOINT="$BASE_MODEL/tokenizer/tokenizer.pth"
export BENCH2DEX_TASK03_MANIFEST="$DATA_ROOT/manifest.json"
export BENCH2DEX_TASK03_EMBEDDING="$DATA_ROOT/t5-embedding.npz"
export BENCH2DEX_ROBOT_MAP="$ROOT/Bench2Dex/robots/active_dof_maps.yml"
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"
export PYTHONUNBUFFERED=1

python="$ENV_PREFIX/bin/python"
test -x "$ENV_PREFIX/bin/torchrun"
test -f "$BENCH2DEX_TASK03_MANIFEST"
test -f "$BENCH2DEX_TASK03_EMBEDDING"
test -f "$BENCH2DEX_ROBOT_MAP"
test -f "$COSMOS_POLICY_PRETRAINED_CHECKPOINT"
test -f "$COSMOS_POLICY_TOKENIZER_CHECKPOINT"
test -d "$INITIAL_CHECKPOINT"
test -d "$REPO"

# The remote shell is noninteractive, so expose the CUDA libraries bundled in
# this environment (including libcudart and NVRTC) without relying on ldconfig.
nvidia_libs="$($python -c 'import nvidia, pathlib; root=pathlib.Path(nvidia.__file__).parent; print(":".join(str(p / "lib") for p in root.iterdir() if (p / "lib").is_dir()))')"
test -n "$nvidia_libs"
export LD_LIBRARY_PATH="$nvidia_libs:${LD_LIBRARY_PATH:-}"

mkdir -p "$TASK03"
if [[ -e "$RUN_DIR" && ! -f "$RUN_DIR/checkpoints/latest_checkpoint.txt" ]]; then
  echo "Run directory exists without a completed resumable checkpoint: $RUN_DIR" >&2
  exit 2
fi

cd "$REPO"
exec "$ENV_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=29617 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py \
  -- \
  "experiment=cosmos_predict2_2b_480p_bench2dex_task03" \
  "job.name=$RUN_NAME" \
  "trainer.max_iter=25000" \
  "trainer.grad_accum_iter=2" \
  "dataloader_train.batch_size=1" \
  "checkpoint.save_iter=1000" \
  "checkpoint.load_path=$INITIAL_CHECKPOINT" \
  "checkpoint.load_training_state=true" \
  2>&1 | tee -a "$LOG_FILE"
