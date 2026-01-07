#!/bin/bash
set -e

NUM=0
MODEL_NAME="alexnet"
OP="matmul"
OP_NAME="${MODEL_NAME}${OP}_$NUM"

OUT_DIR="data"


CV_STRATEGIES=(
  tile_ppwrprp
  tile_ppwrprp_v
  tile_ppwrprp_vr
)
MM_STRATEGIES=(
  tile_p1_v
  tile_goto
  tile3d
)

if [[ "$OP" == "conv2d" ]]; then
    STRATEGIES=("${CV_STRATEGIES[@]}")
elif [[ "$OP" == "matmul" ]]; then
    STRATEGIES=("${MM_STRATEGIES[@]}")
else
    echo "Unknown operator: $OP"
    exit 1
fi

DISPLAY_ARGS=()

mkdir -p "$OUT_DIR"

echo "Running loop-explore for $OP_NAME..."

# Run loop-explore and build display args
for strat in "${STRATEGIES[@]}"; do
    echo "  Strategy: $strat"

    OUTFILE="data/results_${strat}.csv"

    taskset -c 0 loop-explore \
        --operator "$OP" \
        --backends 'tvm' \
        --strategy "$strat" \
        --op-name "$OP_NAME" \
        --output "$OUTFILE"

    # Strip "tile_" prefix for nicer labels
    LABEL="${strat#tile_}"

    # Build loop-display argument
    DISPLAY_ARGS+=("${OUTFILE}:${LABEL}:X:peak:tvm")
done

# Call loop-display with dynamically built args
loop-display \
    --title "Tiling Strategies for ${MODEL_NAME^} ${OP^}$NUM" \
    "${DISPLAY_ARGS[@]}" \
    --output display.svg

