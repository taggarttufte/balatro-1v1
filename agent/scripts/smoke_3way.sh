#!/usr/bin/env bash
# smoke_3way.sh -- the lever comparison TRAIN_NOTES sec.7.3 needs finishing.
#
# WHY: with `--max-ante 4` a cold net converges on skipping 90-99% of Small/Big blinds and
# it is RIGHT to -- it cannot clear an ante-3 Big blind, so playing one costs a life while
# skipping costs a tag's worth of tempo. Scripted anchors fix the TARGET (the net's mean rank
# sits at 0.3-0.5 against their 0.9, so the value head is told it is losing) but not the
# POLICY. Two levers address the actual cause, which is that the net cannot play a blind yet:
#
#   (a) STAGE A WARM-UP  -- train on VANILLA first, where a failed blind is GAME_OVER, then
#                           `--init` those weights into the MLB tournament run.
#   (b) SKIP CAP         -- `--max-skips-per-ante 1`, a training-time mask on the candidate
#                           set (NOT an engine rule), annealed off once the rolling
#                           blind-clear rate passes `--skip-cap-anneal-clear-rate`.
#   (c) BOTH.
#
# Measured already (2026-08-22, RTX 3080 Ti): cold + anchors reaches 93-99% skip by
# generation 4; (a) with only a TEN-MINUTE Stage A held 65-77% flat over 5 generations.
# (b) and (c) are unmeasured -- that is what this script is for.
#
# Runtime: 10 min of Stage A + 3 x 10 min of Stage B = ~40 minutes on one GPU. Raise
# STAGE_A_MIN to 240-360 for a real warm-up; 10 minutes is a smoke and its blind-clear rate
# is still ~0.05.
#
#   bash mp/agent/scripts/smoke_3way.sh                # from the repo root
#   STAGE_A_MIN=240 STAGE_B_MIN=30 bash mp/agent/scripts/smoke_3way.sh
#
# Read the result with:
#   python mp/agent/scripts/smoke_3way_report.py
set -euo pipefail

RUNS=${RUNS:-mp/agent/runs}
DEVICE=${DEVICE:-cuda}
ENCODER=${ENCODER:-mlb}          # use the SAME encoder in both stages: --init refuses a mismatch
SIMS=${SIMS:-40}
STAGE_A_MIN=${STAGE_A_MIN:-10}
STAGE_B_MIN=${STAGE_B_MIN:-10}
TAG=${TAG:-3way}

COMMON=(--device "$DEVICE" --encoder "$ENCODER" --sims "$SIMS" --leaf-batch 1
        --n-agents 16 --seeds-per-gen 2 --max-ante 4 --anchors 0.25 --run-dir "$RUNS")

echo "=== Stage A: vanilla warm-up, ${STAGE_A_MIN} min ==="
python mp/agent/scripts/train_cold.py \
    --minutes "$STAGE_A_MIN" --device "$DEVICE" --ruleset vanilla --encoder "$ENCODER" \
    --sims "$SIMS" --max-decisions 1500 --checkpoint-every 50 \
    --run-dir "$RUNS" --run-name "${TAG}_stageA"
A="$RUNS/${TAG}_stageA/latest.pt"

echo "=== (a) warm start, no skip cap ==="
python mp/agent/scripts/train_mlb.py "${COMMON[@]}" \
    --minutes "$STAGE_B_MIN" --init "$A" --run-name "${TAG}_a"

echo "=== (b) cold, skip cap 1/ante ==="
python mp/agent/scripts/train_mlb.py "${COMMON[@]}" \
    --minutes "$STAGE_B_MIN" --max-skips-per-ante 1 --run-name "${TAG}_b"

echo "=== (c) warm start + skip cap 1/ante ==="
python mp/agent/scripts/train_mlb.py "${COMMON[@]}" \
    --minutes "$STAGE_B_MIN" --init "$A" --max-skips-per-ante 1 --run-name "${TAG}_c"

echo
echo "=== report ==="
python mp/agent/scripts/smoke_3way_report.py --runs "$RUNS" --tag "$TAG"
