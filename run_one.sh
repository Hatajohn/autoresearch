#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_one.sh — launch a single training experiment
#
# Usage:
#   bash run_one.sh <run_num> <pc_weight> <pc_alpha> <time_budget_secs>
#
# Example:
#   bash run_one.sh 11 0.1 0.0 480
#
# What it does:
#   1. Refuses to start if train.py is already running
#   2. Cleans up any orphaned compile_workers from a prior interrupted run
#   3. Runs train.py with the given hyperparameters, streaming output live
#   4. Saves the log to sessions/run<N>_pc_w<W>_a<A>.log
#   5. Prints a short metric summary when done
#
# It does NOT commit, push, sample, or chain into another run.
# Use record_results.sh after reviewing the log to capture results.
# ---------------------------------------------------------------------------
set -euo pipefail

if [ $# -ne 4 ]; then
    echo "Usage: bash run_one.sh <run_num> <pc_weight> <pc_alpha> <time_budget_secs>" >&2
    echo "Example: bash run_one.sh 11 0.1 0.0 480" >&2
    exit 1
fi

RUN_NUM="$1"
PC_WEIGHT="$2"
PC_ALPHA="$3"
TIME_BUDGET="$4"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate

LOGFILE="sessions/run${RUN_NUM}_pc_w${PC_WEIGHT}_a${PC_ALPHA}.log"

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
echo "[run_one] Run ${RUN_NUM}: PC_WEIGHT=${PC_WEIGHT}  PC_ALPHA=${PC_ALPHA}  TIME=${TIME_BUDGET}s"
echo "[run_one] Log: ${LOGFILE}"

# Hard stop if training is already running — we know the state, no need to guess
if pgrep -f "train.py" > /dev/null 2>&1; then
    echo "[run_one] ERROR: train.py is already running. Stop it first:" >&2
    pgrep -a -f "train.py" >&2
    exit 1
fi

# Clean up orphaned compile workers from a previous interrupted run
if pgrep -f "compile_worker" > /dev/null 2>&1; then
    echo "[run_one] Cleaning up orphaned compile_workers..."
    pkill -TERM -f "compile_worker" 2>/dev/null || true
    sleep 2
    pkill -KILL -f "compile_worker" 2>/dev/null || true
fi

echo "[run_one] GPU [before]:"
nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used \
           --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{printf "  temp=%s°C  power=%.0fW  util=%s%%  mem=%sMiB\n",$1,$2,$3,$4}' || echo "  (nvidia-smi unavailable)"

mkdir -p sessions

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "[run_one] Starting at $(date -u '+%Y-%m-%d %H:%M UTC')..."
TRAIN_TIME_BUDGET="$TIME_BUDGET" \
PC_WEIGHT="$PC_WEIGHT" \
PC_ALPHA="$PC_ALPHA" \
    python train.py 2>&1 | tee "$LOGFILE"
EXIT_CODE=${PIPESTATUS[0]}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
if [ $EXIT_CODE -ne 0 ]; then
    echo "[run_one] Training failed (exit ${EXIT_CODE}) — see ${LOGFILE}" >&2
    exit $EXIT_CODE
fi

VAL_BPB=$(grep "val_bpb:"     "$LOGFILE" 2>/dev/null | tail -1 | awk '{print $2}' || echo "N/A")
STEPS=$(  grep "num_steps:"   "$LOGFILE" 2>/dev/null | tail -1 | awk '{print $2}' || echo "N/A")
MFU=$(    grep "mfu_percent:" "$LOGFILE" 2>/dev/null | tail -1 | awk '{print $2}' || echo "N/A")

echo "[run_one] Done — val_bpb=${VAL_BPB}  steps=${STEPS}  mfu=${MFU}%"
echo "[run_one] Review ${LOGFILE}, then:"
echo "[run_one]   bash record_results.sh ${RUN_NUM} ${PC_WEIGHT} ${PC_ALPHA} ${TIME_BUDGET} \"<hypothesis>\" \"<finding>\""
