#!/bin/bash
set -e

# 0. Download checkpoints
mkdir -p checkpoints/dinov2
if [ ! -f checkpoints/sam2_hiera_small.pt ]; then
    echo "Downloading SAM2 Small checkpoint..."
    curl -L -o checkpoints/sam2_hiera_small.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt
fi
if [ ! -f checkpoints/dinov2/dinov2_vits14_pretrain.pth ]; then
    echo "Downloading DinoV2 Small checkpoint..."
    curl -L -o checkpoints/dinov2/dinov2_vits14_pretrain.pth https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
fi

# Define variables
CONFIG=./no_time_to_train/new_exps/coco_fewshot_10shot_Sam2S.yaml
CLASS_SPLIT="few_shot_classes"
RESULTS_DIR=work_dirs/few_shot_results_500
SHOTS=10
SEED=33
GPUS=1 
ACCELERATOR="mps"  # Change to "cuda" if using NVIDIA GPUs

mkdir -p $RESULTS_DIR
FILENAME=few_shot_${SHOTS}shot_seed${SEED}.pkl

# 1. Create reference set
echo "Creating reference set..."
uv run no_time_to_train/dataset/few_shot_sampling.py \
        --n-shot $SHOTS \
        --out-path ${RESULTS_DIR}/${FILENAME} \
        --seed $SEED \
        --dataset $CLASS_SPLIT

# 2. Fill memory with references
echo "Filling memory..."
uv run run_lightning.py test --config $CONFIG \
                              --model.test_mode fill_memory \
                              --out_path ${RESULTS_DIR}/memory.ckpt \
                              --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
                              --model.init_args.dataset_cfgs.fill_memory.memory_pkl ${RESULTS_DIR}/${FILENAME} \
                              --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOTS \
                              --model.init_args.dataset_cfgs.fill_memory.class_split $CLASS_SPLIT \
                              --trainer.logger.save_dir ${RESULTS_DIR}/ \
                              --trainer.devices $GPUS \
                              --trainer.accelerator $ACCELERATOR

# 3. Post-process memory bank
echo "Post-processing memory..."
uv run run_lightning.py test --config $CONFIG \
                              --model.test_mode postprocess_memory \
                              --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
                              --ckpt_path ${RESULTS_DIR}/memory.ckpt \
                              --out_path ${RESULTS_DIR}/memory_postprocessed.ckpt \
                              --trainer.devices 1 \
                              --trainer.accelerator $ACCELERATOR

# 4. Inference on target images (using the 500 image subset)
echo "Running inference on 500 image subset..."
uv run run_lightning.py test --config $CONFIG  \
                              --ckpt_path ${RESULTS_DIR}/memory_postprocessed.ckpt \
                              --model.init_args.test_mode test \
                              --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
                              --model.init_args.model_cfg.dataset_name $CLASS_SPLIT \
                              --model.init_args.dataset_cfgs.test.class_split $CLASS_SPLIT \
                              --model.init_args.dataset_cfgs.test.json_file ./data/coco/annotations_refsam2/val2017_1000.json \
                              --trainer.logger.save_dir ${RESULTS_DIR}/ \
                              --trainer.devices $GPUS \
                              --trainer.accelerator $ACCELERATOR

echo "Done! Results are in $RESULTS_DIR"
