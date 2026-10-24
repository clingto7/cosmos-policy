#!/usr/bin/env bash
set -Eeuo pipefail

readonly DATA_ROOT="/inspire/hdd2/project/roboticsystem2/ky26063/bench2dex_stuff/hqb/task03"
readonly RUN_DIR="$DATA_ROOT/output-budget-effbatch2-25k/cosmos_policy/bench2dex/task03_budget_effbatch2_25k_20261022_r2"
readonly CHECKPOINT_ROOT="$RUN_DIR/checkpoints"
readonly WRAPPER_PID="${1:?pass the training wrapper PID}"
readonly LOG_FILE="$DATA_ROOT/checkpoint-retention-task03-20k.log"
last_pruned_checkpoint=""

prune_verified_checkpoint() {
  [[ -f "$CHECKPOINT_ROOT/latest_checkpoint.txt" ]] || return 0
  local latest_checkpoint latest_digits latest_iteration latest_dir component complete
  latest_checkpoint="$(<"$CHECKPOINT_ROOT/latest_checkpoint.txt")"
  [[ "$latest_checkpoint" =~ ^iter_([0-9]{9})$ ]] || return 0
  [[ "$latest_checkpoint" != "$last_pruned_checkpoint" ]] || return 0
  latest_digits="${BASH_REMATCH[1]}"
  latest_iteration=$((10#$latest_digits))
  [[ "$latest_iteration" -ge 4000 ]] || return 0
  latest_dir="$CHECKPOINT_ROOT/$latest_checkpoint"
  complete=true
  for component in model optim scheduler trainer; do
    if [[ ! -s "$latest_dir/$component/.metadata" || ! -s "$latest_dir/$component/__0_0.distcp" ]]; then
      complete=false
      break
    fi
  done
  [[ "$complete" == true ]] || return 0

  printf '%s verified %s; pruning older temporary checkpoints\n' "$(date -Is)" "$latest_checkpoint" >> "$LOG_FILE"
  local old_dir old_checkpoint old_digits old_iteration
  for old_dir in "$CHECKPOINT_ROOT"/iter_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]; do
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
      printf '%s removed verified superseded checkpoint %s\n' "$(date -Is)" "$old_checkpoint" >> "$LOG_FILE"
    fi
  done
  last_pruned_checkpoint="$latest_checkpoint"
}

while kill -0 "$WRAPPER_PID" 2>/dev/null; do
  prune_verified_checkpoint
  sleep 30
done
prune_verified_checkpoint
