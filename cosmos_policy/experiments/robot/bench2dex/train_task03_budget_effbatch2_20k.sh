#!/usr/bin/env bash
set -Eeuo pipefail

# Continue the existing task 03 trajectory to the 20k evaluation budget.
# Save every 1k for rolling resume; prune after verifying a newer checkpoint.
# Re-running this launcher uses the same output root; Cosmos DCP resumes from
# its latest completed checkpoint when one exists. Keep RUN_NAME unchanged so
# the existing 3k checkpoint remains in this run's checkpoint directory.

readonly ROOT="/inspire/hdd/project/roboticsystem2/ky26063/bench2dex_stuff"
readonly DATA_ROOT="/inspire/hdd2/project/roboticsystem2/ky26063/bench2dex_stuff/hqb/task03"
readonly REPO="$ROOT/hqb/cosmos4dex"
readonly TASK03="$ROOT/hqb/task03"
readonly ENV_PREFIX="$ROOT/miniconda3/envs/cosmos_policy"
readonly RUN_NAME="task03_budget_effbatch2_25k_20261022_r2"
readonly OUTPUT_ROOT="$DATA_ROOT/output-budget-effbatch2-25k"
readonly LOG_FILE="$DATA_ROOT/train-budget-effbatch2-20k-r3.log"
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

checkpoint_root="$RUN_DIR/checkpoints"
if [[ -f "$checkpoint_root/latest_checkpoint.txt" ]]; then
  latest_checkpoint="$(<"$checkpoint_root/latest_checkpoint.txt")"
  if [[ ! "$latest_checkpoint" =~ ^iter_[0-9]{9}$ ]]; then
    echo "Unexpected latest checkpoint pointer: $latest_checkpoint" >&2
    exit 2
  fi
  latest_dir="$checkpoint_root/$latest_checkpoint"
  for component in model optim scheduler trainer; do
    if [[ ! -s "$latest_dir/$component/.metadata" || ! -s "$latest_dir/$component/__0_0.distcp" ]]; then
      echo "Latest checkpoint is incomplete: $latest_dir/$component" >&2
      exit 2
    fi
  done
fi

cd "$REPO"
"$ENV_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=29617 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py \
  -- \
  "experiment=cosmos_predict2_2b_480p_bench2dex_task03" \
  "job.name=$RUN_NAME" \
  "trainer.max_iter=20000" \
  "trainer.grad_accum_iter=2" \
  "dataloader_train.batch_size=1" \
  "checkpoint.save_iter=1000" \
  "checkpoint.load_path=$INITIAL_CHECKPOINT" \
  "checkpoint.load_training_state=true" \
  2>&1 | tee -a "$LOG_FILE" &
training_pipeline_pid=$!
last_pruned_checkpoint=""

prune_verified_checkpoint() {
  [[ -f "$checkpoint_root/latest_checkpoint.txt" ]] || return 0
  local latest_checkpoint latest_digits latest_iteration latest_dir complete component
  latest_checkpoint="$(<"$checkpoint_root/latest_checkpoint.txt")"
  [[ "$latest_checkpoint" =~ ^iter_([0-9]{9})$ ]] || return 0
  [[ "$latest_checkpoint" != "$last_pruned_checkpoint" ]] || return 0
  latest_digits="${BASH_REMATCH[1]}"
  latest_iteration=$((10#$latest_digits))
  [[ "$latest_iteration" -ge 4000 ]] || return 0
  latest_dir="$checkpoint_root/$latest_checkpoint"
  complete=true
  for component in model optim scheduler trainer; do
    if [[ ! -s "$latest_dir/$component/.metadata" || ! -s "$latest_dir/$component/__0_0.distcp" ]]; then
      complete=false
      break
    fi
  done
  [[ "$complete" == true ]] || return 0

  printf 'Verified checkpoint %s; pruning older temporary checkpoints.\n' "$latest_checkpoint"
  local old_dir old_checkpoint old_digits old_iteration
  for old_dir in "$checkpoint_root"/iter_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]; do
    [[ -d "$old_dir" ]] || continue
    old_checkpoint="${old_dir##*/}"
    [[ "$old_checkpoint" =~ ^iter_([0-9]{9})$ ]] || continue
    old_digits="${BASH_REMATCH[1]}"
    old_iteration=$((10#$old_digits))
    if [[ "$old_checkpoint" != "iter_000003000" && \
          "$old_checkpoint" != "iter_000015000" && \
          "$old_checkpoint" != "iter_000020000" ]] && \
       (( old_iteration < latest_iteration )); then
      rm -rf -- "$old_dir"
      printf 'Removed verified superseded checkpoint %s.\n' "$old_checkpoint"
    fi
  done
  last_pruned_checkpoint="$latest_checkpoint"
}

# Keep the recovery anchor at 3k and the requested 15k/20k budgets. Delete an
# older checkpoint only after latest_checkpoint.txt advances and all four DCP
# components for that new checkpoint are present. Never touch newer partials.
while kill -0 "$training_pipeline_pid" 2>/dev/null; do
  prune_verified_checkpoint
  sleep 30
done

training_status=0
wait "$training_pipeline_pid" || training_status=$?
prune_verified_checkpoint
exit "$training_status"
