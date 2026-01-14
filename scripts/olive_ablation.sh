#! /bin/bash

# Run Ablation Study for Olive Dataset
# Shots: 1, 2, 3, 5, 10
# Models: dinov2 ("olive_fewshot_Sam2L.yaml"), dinov3 ("olive_fewshot_Sam2L_dinov3.yaml")

# Define the list of shots
SHOTS_LIST=(1 2 3 5 10)

# Define the model versions
VERSIONS=("dinov2" "dinov3")

# Common settings
CLASS_SPLIT="olive_diseases"
BASE_RESULTS_DIR="work_dirs/olive_ablation"
SEED=42
GPUS=1

export PYTHONPATH=$PYTHONPATH:.

# Loop over model versions
for VERSION in "${VERSIONS[@]}"; do
    if [ "$VERSION" == "dinov2" ]; then
        CONFIG="./no_time_to_train/new_exps/olive_fewshot_Sam2L.yaml"
    elif [ "$VERSION" == "dinov3" ]; then
        CONFIG="./no_time_to_train/new_exps/olive_fewshot_Sam2L_dinov3.yaml"
    fi

    # Loop over shots
    for SHOTS in "${SHOTS_LIST[@]}"; do
        echo "========================================================"
        echo "Running ablation for Version: $VERSION with Shots: $SHOTS"
        echo "Config: $CONFIG"
        echo "========================================================"
        
        RESULTS_DIR="${BASE_RESULTS_DIR}/${VERSION}_${SHOTS}shot"
        mkdir -p $RESULTS_DIR
        
        FILENAME=olive_${SHOTS}shot_seed${SEED}.pkl

        echo "Sampling few-shot support set..."
        python no_time_to_train/dataset/few_shot_sampling.py \
                --n-shot $SHOTS \
                --out-path ${RESULTS_DIR}/${FILENAME} \
                --seed $SEED \
                --dataset $CLASS_SPLIT \
                --dataset-json ./data/olive_diseases/annotations/instances_train2017.json

        echo "Filling memory bank..."
        python run_lightning.py test --config $CONFIG \
                                      --model.test_mode fill_memory \
                                      --out_path ${RESULTS_DIR}/memory.ckpt \
                                      --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
                                      --model.init_args.dataset_cfgs.fill_memory.memory_pkl ${RESULTS_DIR}/${FILENAME} \
                                      --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOTS \
                                      --model.init_args.dataset_cfgs.fill_memory.class_split $CLASS_SPLIT \
                                      --model.init_args.model_cfg.dataset_name $CLASS_SPLIT \
                                      --trainer.logger.save_dir ${RESULTS_DIR}/ \
                                      --trainer.devices $GPUS

        echo "Post-processing memory bank..."
        python run_lightning.py test --config $CONFIG \
                                      --model.test_mode postprocess_memory \
                                      --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
                                      --model.init_args.model_cfg.dataset_name $CLASS_SPLIT \
                                      --ckpt_path ${RESULTS_DIR}/memory.ckpt \
                                      --out_path ${RESULTS_DIR}/memory_postprocessed.ckpt \
                                      --trainer.devices 1

        echo "Running testing..."
        python run_lightning.py test --config $CONFIG  \
                                      --ckpt_path ${RESULTS_DIR}/memory_postprocessed.ckpt \
                                      --model.init_args.test_mode test \
                                      --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
                                      --model.init_args.model_cfg.dataset_name $CLASS_SPLIT \
                                      --model.init_args.dataset_cfgs.test.class_split $CLASS_SPLIT \
                                      --trainer.logger.save_dir ${RESULTS_DIR}/ \
                                      --trainer.devices $GPUS \
                                      --seed $SEED \
                                      --n_shot $SHOTS \
                                      --export_result ${RESULTS_DIR}/results.json

        echo "Cleaning up checkpoints..."
        rm ${RESULTS_DIR}/*.ckpt
                                      
    done
done
