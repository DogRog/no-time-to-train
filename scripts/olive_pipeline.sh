#! /bin/bash

# GPUS is the number of GPUs to use
GPUS=${1:-1}
# SHOTS can be an array of numbers
SHOTS=(1 5)
# SEED is the seed to use for the random number generator
SEED=42
# YAML file to use for the config
YAML_FILE=./no_time_to_train/new_exps/olive_fewshot.yaml

# Olive diseases has 6 classes
CATEGORY_NUM=6

DATASET_NAME="olive_diseases"
TRAIN_JSON="./data/olive_diseases/train/_annotations.coco.json"
TRAIN_IMG_DIR="./data/olive_diseases/train"
VALID_JSON="./data/olive_diseases/valid/_annotations.coco.json"
VALID_IMG_DIR="./data/olive_diseases/valid"
SAM_CHECKPOINT="./checkpoints/sam_vit_h_4b8939.pth"

# 1. Convert Bounding Boxes to Segmentation Masks using SAM
# This generates _with_segm.json files
echo "Converting Train Annotations..."
if [ ! -f "${TRAIN_JSON%.json}_with_segm.json" ]; then
    python no_time_to_train/dataset/sam_bbox_to_segm_batch.py \
        --input_json $TRAIN_JSON \
        --image_dir $TRAIN_IMG_DIR \
        --sam_checkpoint $SAM_CHECKPOINT \
        --model_type vit_h \
        --device cuda
else
    echo "Train segmentation json already exists. Skipping."
fi

echo "Converting Valid Annotations..."
if [ ! -f "${VALID_JSON%.json}_with_segm.json" ]; then
    python no_time_to_train/dataset/sam_bbox_to_segm_batch.py \
        --input_json $VALID_JSON \
        --image_dir $VALID_IMG_DIR \
        --sam_checkpoint $SAM_CHECKPOINT \
        --model_type vit_h \
        --device cuda
else
    echo "Valid segmentation json already exists. Skipping."
fi

TRAIN_JSON_SEGM="${TRAIN_JSON%.json}_with_segm.json"
# VALID_JSON_SEGM="${VALID_JSON%.json}_with_segm.json"


for SHOT in "${SHOTS[@]}"; do
    echo -e "\033[31mOLIVE DISEASES PIPELINE, SHOT $SHOT\033[0m"

    RESULTS_DIR=work_dirs/olive_diseases_seed${SEED}/${SHOT}shot
    if [ ! -d "$RESULTS_DIR" ]; then
        mkdir -p $RESULTS_DIR
    fi
    FILENAME=olive_few_shot_ann_${SHOT}shot_seed${SEED}.pkl

    # 2. Sample Memory (Reference) Set
    # Uses the segmented train annotation
    python no_time_to_train/dataset/few_shot_sampling.py --n-shot $SHOT \
                                                --out-path $RESULTS_DIR/${FILENAME} \
                                                --seed $SEED \
                                                --dataset $DATASET_NAME \
                                                --img-dir $TRAIN_IMG_DIR \
                                                --dataset-json $TRAIN_JSON_SEGM \
                                                --plot

    # 3. Reference Feature Extraction (Fill Memory)
    python run_lightning.py test --config $YAML_FILE \
                                --model.test_mode fill_memory \
                                --out_path $RESULTS_DIR/memory.ckpt \
                                --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
                                --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
                                --model.init_args.dataset_cfgs.fill_memory.memory_pkl $RESULTS_DIR/${FILENAME} \
                                --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOT \
                                --model.init_args.dataset_cfgs.fill_memory.class_split $DATASET_NAME \
                                --trainer.logger.save_dir $RESULTS_DIR/ \
                                --trainer.devices $GPUS

    # 4. Post-process Memory
    python run_lightning.py test --config $YAML_FILE \
                                --model.test_mode postprocess_memory \
                                --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
                                --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
                                --ckpt_path $RESULTS_DIR/memory.ckpt \
                                --out_path $RESULTS_DIR/memory_postprocessed.ckpt \
                                --trainer.logger.save_dir $RESULTS_DIR/ \
                                --trainer.devices 1

    # 5. Run Inference (Test)
    python run_lightning.py test --config $YAML_FILE  \
                                --ckpt_path $RESULTS_DIR/memory_postprocessed.ckpt \
                                --model.init_args.test_mode test \
                                --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
                                --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
                                --model.init_args.dataset_cfgs.test.class_split $DATASET_NAME \
                                --trainer.logger.save_dir $RESULTS_DIR/ \
                                --trainer.devices $GPUS \
                                --n_shot $SHOT \
                                --seed $SEED
done
