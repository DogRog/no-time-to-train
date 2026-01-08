#! /bin/bash

# Dowload olive dataset
python 'no_time_to_train/dataset/download_dataset.py' --dataset-name olive

# Convert olive dataset into segmentation format
python no_time_to_train/dataset/sam_bbox_to_segm_batch.py \
    --input_json data/olive_diseases/train/_annotations.coco.json \
    --image_dir data/olive_diseases/train \
    --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
    --model_type vit_h

python no_time_to_train/dataset/sam_bbox_to_segm_batch.py \
    --input_json data/olive_diseases/valid/_annotations.coco.json \
    --image_dir data/olive_diseases/valid \
    --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
    --model_type vit_h

python no_time_to_train/dataset/sam_bbox_to_segm_batch.py \
    --input_json data/olive_diseases/test/_annotations.coco.json \
    --image_dir data/olive_diseases/test \
    --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
    --model_type vit_h

# Organize into COCO structure
echo "Organizing into COCO structure..."
mkdir -p data/olive_diseases/annotations
mkdir -p data/olive_diseases/train2017
mkdir -p data/olive_diseases/val2017
mkdir -p data/olive_diseases/test2017

mv data/olive_diseases/train/_annotations.coco_with_segm.json data/olive_diseases/annotations/instances_train2017.json
mv data/olive_diseases/valid/_annotations.coco_with_segm.json data/olive_diseases/annotations/instances_val2017.json
mv data/olive_diseases/test/_annotations.coco_with_segm.json data/olive_diseases/annotations/instances_test2017.json

mv data/olive_diseases/train/*.jpg data/olive_diseases/train2017/
mv data/olive_diseases/valid/*.jpg data/olive_diseases/val2017/
mv data/olive_diseases/test/*.jpg data/olive_diseases/test2017/

# Remove old directories
rm -rf data/olive_diseases/train
rm -rf data/olive_diseases/valid
rm -rf data/olive_diseases/test