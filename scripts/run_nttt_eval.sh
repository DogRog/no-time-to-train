#! /bin/bash
set -euo pipefail

# NTTT-only evaluation on Olive dataset.
# Exports predictions to COCO-style JSON for later metric computation.

CONFIG_FILE=${CONFIG_FILE:-scripts/config/olive_eval.conf}
if [[ -f "${CONFIG_FILE}" ]]; then
    source "${CONFIG_FILE}"
fi

SEED=${SEED:-42}
GPUS=${GPUS:-1}
CONFIG=${CONFIG:-./no_time_to_train/new_exps/olive_fewshot_Sam2L.yaml}
CLASS_SPLIT=${CLASS_SPLIT:-olive_diseases}
OUTPUT_ROOT=${OUTPUT_ROOT:-work_dirs/olive_nttt_sam3_eval}

TRAIN_JSON=${TRAIN_JSON:-./data/olive_diseases/annotations/instances_train2017.json}
VAL_JSON=${VAL_JSON:-./data/olive_diseases/annotations/instances_val2017.json}

export PYTHONPATH=${PYTHONPATH:-}:.

NVIDIA_SMI_BIN=$(command -v nvidia-smi || true)
GPU_MONITOR_PID=""

start_gpu_monitor() {
    local sample_file="$1"
    if [[ -z "${NVIDIA_SMI_BIN}" ]]; then
        GPU_MONITOR_PID=""
        return
    fi

    : > "${sample_file}"
    (
        while true; do
            "${NVIDIA_SMI_BIN}" --query-gpu=memory.used --format=csv,noheader,nounits \
                | awk 'BEGIN {max=0} {if ($1 > max) max=$1} END {print max}' >> "${sample_file}" 2>/dev/null || true
            sleep 1
        done
    ) >/dev/null 2>&1 &
    GPU_MONITOR_PID="$!"
}

stop_gpu_monitor() {
    local monitor_pid="$1"
    local sample_file="$2"

    if [[ -n "${monitor_pid}" ]]; then
        kill "${monitor_pid}" >/dev/null 2>&1 || true
        wait "${monitor_pid}" 2>/dev/null || true
    fi

    if [[ -f "${sample_file}" ]]; then
        awk 'BEGIN {max=0} {if ($1 > max) max=$1} END {print max}' "${sample_file}"
    else
        echo "nan"
    fi
}

NUM_VAL_IMAGES=$(python - <<PY
import json
with open("${VAL_JSON}", "r") as f:
    data = json.load(f)
print(len(data.get("images", [])))
PY
)

echo "Validation images for FPS computation: ${NUM_VAL_IMAGES}"

if [[ -n "${SHOTS_CSV:-}" ]]; then
    IFS=',' read -r -a SHOTS_LIST <<< "${SHOTS_CSV}"
else
    SHOTS_LIST=(1 2 3 5 10)
fi

if [[ -n "${VERSIONS_CSV:-}" ]]; then
    IFS=',' read -r -a VERSIONS <<< "${VERSIONS_CSV}"
else
    VERSIONS=("dinov2_small" "dinov2_base" "dinov2_large" "dinov2_giant" "dinov3_small" "dinov3_base" "dinov3_large" "dinov3_huge")
fi

for VERSION in "${VERSIONS[@]}"; do
    for SHOTS in "${SHOTS_LIST[@]}"; do
        RUN_DIR=${OUTPUT_ROOT}/${VERSION}_${SHOTS}shot_seed${SEED}
        mkdir -p "${RUN_DIR}"

        FEWSHOT_PKL=olive_${SHOTS}shot_seed${SEED}.pkl
        NTTT_PRED_JSON=${RUN_DIR}/nttt_predictions.json

        echo "========================================================"
        echo "Running NTTT evaluation"
        echo "Shots: ${SHOTS} | Seed: ${SEED} | Encoder: ${VERSION}"
        echo "Output dir: ${RUN_DIR}"
        echo "========================================================"

        echo "[1/4] Sampling few-shot support set..."
        python no_time_to_train/dataset/few_shot_sampling.py \
            --n-shot "${SHOTS}" \
            --out-path "${RUN_DIR}/${FEWSHOT_PKL}" \
            --seed "${SEED}" \
            --dataset "${CLASS_SPLIT}" \
            --dataset-json "${TRAIN_JSON}"

        echo "[2/4] Filling NTTT memory bank..."
        python run_lightning.py test --config "${CONFIG}" \
            --model.test_mode fill_memory \
            --out_path "${RUN_DIR}/memory.ckpt" \
            --model.init_args.model_cfg.encoder_cfg "${VERSION}" \
            --model.init_args.model_cfg.memory_bank_cfg.length "${SHOTS}" \
            --model.init_args.dataset_cfgs.fill_memory.memory_pkl "${RUN_DIR}/${FEWSHOT_PKL}" \
            --model.init_args.dataset_cfgs.fill_memory.memory_length "${SHOTS}" \
            --model.init_args.dataset_cfgs.fill_memory.class_split "${CLASS_SPLIT}" \
            --model.init_args.model_cfg.dataset_name "${CLASS_SPLIT}" \
            --trainer.logger.save_dir "${RUN_DIR}/" \
            --trainer.devices "${GPUS}"

        echo "[3/4] Post-processing NTTT memory bank..."
        python run_lightning.py test --config "${CONFIG}" \
            --model.test_mode postprocess_memory \
            --model.init_args.model_cfg.encoder_cfg "${VERSION}" \
            --model.init_args.model_cfg.memory_bank_cfg.length "${SHOTS}" \
            --model.init_args.model_cfg.dataset_name "${CLASS_SPLIT}" \
            --ckpt_path "${RUN_DIR}/memory.ckpt" \
            --out_path "${RUN_DIR}/memory_postprocessed.ckpt" \
            --trainer.devices 1

        echo "[4/4] Running NTTT test and exporting predictions..."
        GPU_SAMPLE_FILE="${RUN_DIR}/.nttt_gpu_mem_samples.txt"
        start_gpu_monitor "${GPU_SAMPLE_FILE}"

        TEST_START=$(python - <<'PY'
import time
print(time.perf_counter())
PY
)
        python run_lightning.py test --config "${CONFIG}" \
            --ckpt_path "${RUN_DIR}/memory_postprocessed.ckpt" \
            --model.init_args.test_mode test \
            --model.init_args.model_cfg.encoder_cfg "${VERSION}" \
            --model.init_args.model_cfg.memory_bank_cfg.length "${SHOTS}" \
            --model.init_args.model_cfg.dataset_name "${CLASS_SPLIT}" \
            --model.init_args.dataset_cfgs.test.class_split "${CLASS_SPLIT}" \
            --trainer.logger.save_dir "${RUN_DIR}/" \
            --trainer.devices "${GPUS}" \
            --seed "${SEED}" \
            --n_shot "${SHOTS}" \
            --export_result "${NTTT_PRED_JSON}"

        TEST_END=$(python - <<'PY'
import time
print(time.perf_counter())
PY
)

        NTTT_PEAK_VRAM_MIB=$(stop_gpu_monitor "${GPU_MONITOR_PID}" "${GPU_SAMPLE_FILE}")
        rm -f "${GPU_SAMPLE_FILE}" || true

        TEST_DURATION=$(python - <<PY
start = float("${TEST_START}")
end = float("${TEST_END}")
print(max(0.0, end - start))
PY
)

        NTTT_FPS=$(python - <<PY
num_images = float("${NUM_VAL_IMAGES}")
duration = float("${TEST_DURATION}")
print((num_images / duration) if duration > 0 else 0.0)
PY
)

        python - <<PY
import json

runtime = {
    "model": "nttt",
    "encoder": "${VERSION}",
    "shots": int("${SHOTS}"),
    "seed": int("${SEED}"),
    "num_images": int("${NUM_VAL_IMAGES}"),
    "test_time_sec": float("${TEST_DURATION}"),
    "fps": float("${NTTT_FPS}"),
    "peak_vram_mib": None if "${NTTT_PEAK_VRAM_MIB}" == "nan" else float("${NTTT_PEAK_VRAM_MIB}"),
}
with open("${RUN_DIR}/nttt_runtime.json", "w") as f:
    json.dump(runtime, f, indent=2)
PY
        echo "NTTT FPS: ${NTTT_FPS}"
        echo "NTTT peak VRAM (MiB): ${NTTT_PEAK_VRAM_MIB}"

        rm -f "${RUN_DIR}"/*.ckpt || true

        echo "Done. NTTT predictions saved to: ${NTTT_PRED_JSON}"
    done
done

