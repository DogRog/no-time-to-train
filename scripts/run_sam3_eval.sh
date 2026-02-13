#! /bin/bash
set -euo pipefail

# SAM3-only evaluation on Olive dataset.
# Exports predictions to COCO-style JSON for later metric computation.

CONFIG_FILE=${CONFIG_FILE:-scripts/config/olive_eval.conf}
if [[ -f "${CONFIG_FILE}" ]]; then
    source "${CONFIG_FILE}"
fi

SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda}
SAM3_RUN_PREFIX=${SAM3_RUN_PREFIX:-sam3}
OUTPUT_ROOT=${OUTPUT_ROOT:-work_dirs/olive_nttt_sam3_eval}
PREDICTION_NAME=${PREDICTION_NAME:-sam3_predictions.json}

export PYTHONPATH=${PYTHONPATH:-}:.

if [[ -n "${SHOTS_CSV:-}" ]]; then
    IFS=',' read -r -a SHOTS_LIST <<< "${SHOTS_CSV}"
else
    SHOTS_LIST=(1 2 3 5 10)
fi

for SHOTS in "${SHOTS_LIST[@]}"; do
    RUN_DIR=${OUTPUT_ROOT}/${SAM3_RUN_PREFIX}_${SHOTS}shot_seed${SEED}
    mkdir -p "${RUN_DIR}"

    echo "========================================================"
    echo "Running SAM3 evaluation"
    echo "Shots: ${SHOTS} | Seed: ${SEED} | Device: ${DEVICE}"
    echo "Run folder key: ${SAM3_RUN_PREFIX}"
    echo "Output dir: ${RUN_DIR}"
    echo "========================================================"

    python scripts/eval_sam3_video_olive.py \
        --shots "${SHOTS}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --output_dir "${RUN_DIR}" \
        --prediction_file "${PREDICTION_NAME}" \
        --evaluate_coco

    echo "Done. SAM3 predictions saved to: ${RUN_DIR}/${PREDICTION_NAME}"
done
