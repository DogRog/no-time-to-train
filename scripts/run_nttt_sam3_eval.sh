#! /bin/bash
set -euo pipefail

# Combined SAM3 + NTTT evaluation launcher.
# Reuses scripts/run_sam3_eval.sh and scripts/run_nttt_eval.sh.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

CONFIG_FILE=${CONFIG_FILE:-${SCRIPT_DIR}/config/olive_eval.conf}
RUN_SAM3=${RUN_SAM3:-1}
RUN_NTTT=${RUN_NTTT:-1}

SAM3_SCRIPT=${SAM3_SCRIPT:-${SCRIPT_DIR}/run_sam3_eval.sh}
NTTT_SCRIPT=${NTTT_SCRIPT:-${SCRIPT_DIR}/run_nttt_eval.sh}

if [[ "${RUN_SAM3}" != "1" && "${RUN_NTTT}" != "1" ]]; then
    echo "Nothing to run: both RUN_SAM3 and RUN_NTTT are disabled."
    exit 1
fi

if [[ ! -f "${SAM3_SCRIPT}" ]]; then
    echo "SAM3 script not found: ${SAM3_SCRIPT}"
    exit 1
fi

if [[ ! -f "${NTTT_SCRIPT}" ]]; then
    echo "NTTT script not found: ${NTTT_SCRIPT}"
    exit 1
fi

if [[ -f "${CONFIG_FILE}" ]]; then
    set -a
    source "${CONFIG_FILE}"
    set +a
fi

export CONFIG_FILE
export PYTHONPATH=${PYTHONPATH:-}:.

cd "${REPO_ROOT}"

echo "========================================================"
echo "Combined Olive evaluation launcher"
echo "Config file: ${CONFIG_FILE}"
echo "Run SAM3: ${RUN_SAM3} | Run NTTT: ${RUN_NTTT}"
echo "Output root: ${OUTPUT_ROOT:-work_dirs/olive_nttt_sam3_eval}"
echo "========================================================"

if [[ "${RUN_SAM3}" == "1" ]]; then
    echo "[1/2] Launching SAM3 evaluation..."
    bash "${SAM3_SCRIPT}"
else
    echo "[1/2] Skipping SAM3 evaluation (RUN_SAM3=${RUN_SAM3})."
fi

if [[ "${RUN_NTTT}" == "1" ]]; then
    echo "[2/2] Launching NTTT evaluation..."
    bash "${NTTT_SCRIPT}"
else
    echo "[2/2] Skipping NTTT evaluation (RUN_NTTT=${RUN_NTTT})."
fi

echo "All requested evaluations completed."
