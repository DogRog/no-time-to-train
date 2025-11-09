#!/usr/bin/env bash

CONFIG=./no_time_to_train/new_exps/coco_fewshot_10shot_Sam2L_dinov3_large.yaml
CLASS_SPLIT="few_shot_classes"
RESULTS_DIR=work_dirs/few_shot_results
SHOTS=30
SEED=33
GPUS=1

mkdir -p $RESULTS_DIR
FILENAME=few_shot_${SHOTS}shot_seed${SEED}.pkl

python no_time_to_train/dataset/few_shot_sampling.py \
        --n-shot $SHOTS \
        --out-path ${RESULTS_DIR}/${FILENAME} \
        --seed $SEED \
        --dataset $CLASS_SPLIT

python run_lightening.py test --config $CONFIG \
                              --model.test_mode fill_memory \
                              --out_path ${RESULTS_DIR}/memory.ckpt \
                              --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
                              --model.init_args.dataset_cfgs.fill_memory.memory_pkl ${RESULTS_DIR}/${FILENAME} \
                              --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOTS \
                              --model.init_args.dataset_cfgs.fill_memory.class_split $CLASS_SPLIT \
                              --trainer.logger.save_dir ${RESULTS_DIR}/ \
                              --trainer.devices $GPUS

# python run_lightening.py test --config $CONFIG \
#                               --model.test_mode postprocess_memory \
#                               --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
#                               --ckpt_path ${RESULTS_DIR}/memory.ckpt \
#                               --out_path ${RESULTS_DIR}/memory_postproceassed.ckpt \
#                               --trainer.devices 1


# python run_lightening.py test --config $CONFIG  \
#                               --ckpt_path ${RESULTS_DIR}/memory_postprocessed.ckpt \
#                               --model.init_args.test_mode test \
#                               --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
#                               --model.init_args.model_cfg.dataset_name $CLASS_SPLIT \
#                               --model.init_args.dataset_cfgs.test.class_split $CLASS_SPLIT \
#                               --trainer.logger.save_dir ${RESULTS_DIR}/ \
#                               --trainer.devices $GPUS