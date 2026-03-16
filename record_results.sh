#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# record_results.sh — capture results from a completed run into session_pc.md
#
# Run this MANUALLY after reviewing a log produced by run_one.sh.
#
# Usage:
#   bash record_results.sh <run_num> <pc_weight> <time_budget_secs> \
#       "<hypothesis>" "<finding>"
#
# The hypothesis and finding are free-text strings you write yourself after
# looking at the log.  If omitted they default to placeholder text.
#
# Example:
#   bash record_results.sh 15 0.1 600 \
#       "Broadcast PC baseline: does pc3 match prior val_bpb?" \
#       "CONFIRMED: val_bpb=1.19xx, broadcast PC overhead acceptable"
#
# What it does:
#   1. Extracts metrics from the log
#   2. Generates 3 samples from the current checkpoint.pt
#   3. Appends a structured block to sessions/session_pc.md
#
# It does NOT commit or push — do that yourself after reviewing the output.
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
if [ $# -lt 3 ]; then
    echo "Usage: bash record_results.sh <run_num> <pc_weight> <time_budget_secs> [hypothesis] [finding]" >&2
    exit 1
fi

RUN_NUM="$1"
PC_WEIGHT="$2"
TIME_BUDGET="$3"
HYPOTHESIS="${4:-[fill in hypothesis]}"
FINDING="${5:-[fill in finding after reviewing log]}"
PC_FOCAL_GAMMA="${PC_FOCAL_GAMMA:-1.0}"
KL_WEIGHT="${KL_WEIGHT:-0.01}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate

LOGFILE="sessions/run${RUN_NUM}_pc3_w${PC_WEIGHT}.log"
SESSION_FILE="sessions/session_pc.md"

if [ ! -f "$LOGFILE" ]; then
    echo "[record] ERROR: log file not found: ${LOGFILE}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Extract metrics
# ---------------------------------------------------------------------------
VAL_BPB=$(grep "val_bpb:" "$LOGFILE" 2>/dev/null | tail -1 | awk '{print $2}' || true)
STEPS=$(grep "num_steps:" "$LOGFILE" 2>/dev/null | tail -1 | awk '{print $2}' || true)
MFU=$(grep "mfu_percent:" "$LOGFILE" 2>/dev/null | tail -1 | awk '{print $2}' || true)
TRAIN_SECS=$(grep "training_seconds:" "$LOGFILE" 2>/dev/null | tail -1 | awk '{print $2}' || true)
FINAL_PC_LOSS=$(grep "^step" "$LOGFILE" 2>/dev/null | tail -1 \
    | grep -oP 'pc_loss: \K[0-9.]+' || true)

VAL_BPB="${VAL_BPB:-N/A}"
STEPS="${STEPS:-N/A}"
MFU="${MFU:-N/A}"
TRAIN_SECS="${TRAIN_SECS:-N/A}"
FINAL_PC_LOSS="${FINAL_PC_LOSS:-N/A}"

echo "[record] Metrics from ${LOGFILE}:"
echo "  val_bpb=${VAL_BPB}  steps=${STEPS}  mfu=${MFU}%  train_secs=${TRAIN_SECS}  pc_loss_final=${FINAL_PC_LOSS}"

# Compute delta vs baseline
BASELINE=1.183689
VS_BASELINE=$(python -c "
v = '${VAL_BPB}'
if v == 'N/A':
    print('N/A')
else:
    print(f'{float(v) - ${BASELINE}:+.6f}')
" 2>/dev/null || echo "?")

# Warn if too few steps for meaningful val_bpb
if [ "$STEPS" != "N/A" ] && [ "$STEPS" -lt 20 ] 2>/dev/null; then
    echo "[record] WARNING: only ${STEPS} steps — val_bpb may not be meaningful." >&2
fi

# ---------------------------------------------------------------------------
# Generate samples
# ---------------------------------------------------------------------------
echo "[record] Generating 3 samples from checkpoint.pt..."
SAMPLES=""
if [ -f "checkpoint.pt" ]; then
    for i in 1 2 3; do
        SAMPLE=$(python sample.py --tokens 150 --temp 0.8 2>/dev/null \
            | grep -A 30 "prompt: ''" || true)
        SAMPLES="${SAMPLES}--- sample ${i} ---
${SAMPLE}

"
    done
else
    SAMPLES="[no checkpoint.pt found]"
fi

# ---------------------------------------------------------------------------
# Append to session_pc.md
# ---------------------------------------------------------------------------
cat >> "$SESSION_FILE" <<MDBLOCK

---

### Run ${RUN_NUM} — $(date -u '+%Y-%m-%d %H:%M UTC')

| Field | Value |
|---|---|
| **val_bpb** | ${VAL_BPB} (${VS_BASELINE} vs baseline 1.183689) |
| **PC_WEIGHT** | ${PC_WEIGHT} |
| **PC_FOCAL_GAMMA** | ${PC_FOCAL_GAMMA} |
| **KL_WEIGHT** | ${KL_WEIGHT} |
| **TIME_BUDGET** | ${TIME_BUDGET}s |
| **Steps** | ${STEPS} |
| **Training seconds** | ${TRAIN_SECS} |
| **MFU** | ${MFU}% |
| **Final pc_loss** | ${FINAL_PC_LOSS} |
| **Log** | ${LOGFILE} |

**Hypothesis:** ${HYPOTHESIS}

**Finding:** ${FINDING}

#### Samples — Run ${RUN_NUM}

\`\`\`
${SAMPLES}
\`\`\`
MDBLOCK

echo "[record] Appended Run ${RUN_NUM} to ${SESSION_FILE}"
echo "[record] Review the entry, then commit manually:"
echo "[record]   git add sessions/ && git commit -m 'Run ${RUN_NUM} results'"
