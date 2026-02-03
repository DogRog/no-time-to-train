import argparse
import json
import os
import random
import statistics

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 Few-Shot Episodic Evaluation for Olive Diseases")
    parser.add_argument("--coco_json", type=str, default="data/olive_diseases/annotations/instances_all.json", help="Path to COCO JSON annotations")
    parser.add_argument("--img_dir", type=str, default="data/olive_diseases/all_images", help="Directory with all images")
    parser.add_argument("--checkpoint", type=str, default="facebook/sam3", help="SAM3 Model ID")
    parser.add_argument("--shots", type=str, default="1,2,3,5,10", help="K-Shots to evaluate")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes per class (N)")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    return parser.parse_args()

# --- Helper: IoU Calculation ---
def calculate_iou(pred_mask, gt_mask):
    """
    Computes binary IoU between two numpy masks (0 or 1).
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0 # Both empty = perfect match
    return intersection / union

# --- Helper: Image & Mask Loading ---
def get_image_and_gt(coco, img_dir, img_id, cat_id):
    """
    Loads image and generates binary Ground Truth mask for the specific category.
    """
    img_info = coco.loadImgs([img_id])[0]
    img_path = os.path.join(img_dir, img_info['file_name'])
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    # Generate GT Binary Mask for this specific class
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        if isinstance(ann['segmentation'], dict): # RLE
            m = mask_utils.decode(ann['segmentation'])
        else: # Polygon
            m = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(ann['segmentation'], h, w)))
        gt_mask = np.maximum(gt_mask, m)
    
    # Get BBoxes for Prompting (if this image is used as Support)
    bboxes = []
    for ann in anns:
        x, y, bw, bh = ann['bbox']
        bboxes.append([x, y, x + bw, y + bh]) # xyxy format

    return image, gt_mask, bboxes

# --- Helper: Visual Prompting (Concatenation) ---
def create_visual_prompt(support_data, query_image, target_h=1024):
    """
    Concatenates Support images (Left) and Query image (Right).
    Adjusts BBoxes to new coordinates.
    """
    resized_supports = []
    support_boxes_shifted = []
    current_x = 0

    # 1. Process Support Images
    for img, boxes in support_data:
        w, h = img.size
        scale = target_h / h
        new_w = int(w * scale)
        img_resized = img.resize((new_w, target_h), Image.Resampling.LANCZOS)
        
        # Shift boxes
        for box in boxes:
            bx1, by1, bx2, by2 = box
            support_boxes_shifted.append([
                bx1 * scale + current_x,
                by1 * scale,
                bx2 * scale + current_x,
                by2 * scale
            ])
        
        resized_supports.append(img_resized)
        current_x += new_w

    # 2. Process Query Image
    qw, qh = query_image.size
    q_scale = target_h / qh
    q_new_w = int(qw * q_scale)
    query_resized = query_image.resize((q_new_w, target_h), Image.Resampling.LANCZOS)
    
    # 3. Stitch Canvas
    total_w = current_x + q_new_w
    canvas = Image.new("RGB", (total_w, target_h))
    
    x_offset = 0
    for simg in resized_supports:
        canvas.paste(simg, (x_offset, 0))
        x_offset += simg.size[0]
    
    # Query is pasted at the end
    query_x_start = x_offset
    canvas.paste(query_resized, (query_x_start, 0))
    
    # Return everything needed
    # query_roi = (x_start, y_start, x_end, y_end) on the canvas
    query_roi = (query_x_start, 0, query_x_start + q_new_w, target_h)
    
    return canvas, support_boxes_shifted, query_roi

# --- Main Pipeline ---
def main():
    args = parse_args()
    
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"--- SAM3 Olive Disease Evaluator ---")
    print(f"Loading Model: {args.checkpoint}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Sam3Model.from_pretrained(args.checkpoint).to(device)
    processor = Sam3Processor.from_pretrained(args.checkpoint)
    model.eval()

    print(f"Loading Dataset: {args.coco_json}...")
    coco = COCO(args.coco_json)
    cat_ids = coco.getCatIds()
    cat_names = {c['id']: c['name'] for c in coco.loadCats(cat_ids)}
    
    shots_list = [int(s) for s in args.shots.split(",")]
    
    # Results Storage
    # Structure: results[shot][class_name] = [iou_1, iou_2, ..., iou_N]
    final_results = {k: {name: [] for name in cat_names.values()} for k in shots_list}

    # --- OUTER LOOP: K-SHOTS ---
    for k in shots_list:
        print(f"\n[Starting Evaluation for K={k} Shots]")
        
        # --- MIDDLE LOOP: CATEGORIES ---
        for cat_id in cat_ids:
            cat_name = cat_names[cat_id]
            img_ids = coco.getImgIds(catIds=[cat_id])
            
            # Skip if not enough images
            if len(img_ids) < k + 1:
                print(f"Skipping {cat_name} (Not enough images for {k}-shot)")
                continue

            print(f"  > Class: {cat_name} | Episodes: {args.episodes}")

            # --- INNER LOOP: EPISODES (The 1000 runs) ---
            # We use tqdm here to track the 1000 episodes
            for episode in tqdm(range(args.episodes), leave=False):
                
                # 1. Sample Support (K) and Query (1)
                # Randomly shuffle all available IDs
                random.shuffle(img_ids)
                
                support_ids = img_ids[:k]
                query_id = img_ids[k] # The (K+1)th image is the query
                
                try:
                    # 2. Prepare Support Data
                    support_data = []
                    for sid in support_ids:
                        s_img, _, s_bboxes = get_image_and_gt(coco, args.img_dir, sid, cat_id)
                        # Filter out images that might have empty annotations (bad data)
                        if len(s_bboxes) > 0:
                            support_data.append((s_img, s_bboxes))
                    
                    if len(support_data) < k:
                        # Fallback if bad data found
                        continue 

                    # 3. Prepare Query Data
                    q_img, q_gt_mask, _ = get_image_and_gt(coco, args.img_dir, query_id, cat_id)

                    # 4. Create Visual Prompt (Stitch)
                    canvas, prompt_boxes, query_roi = create_visual_prompt(support_data, q_img)
                    
                    # 5. Run SAM3
                    # SAM3 Input: Image + Prompt Boxes.
                    # Note: We provide boxes for the support part. SAM3 infers the rest.
                    inputs = processor(
                        images=canvas, 
                        input_boxes=[prompt_boxes], 
                        input_boxes_labels=[[1] * len(prompt_boxes)], # 1 = foreground
                        return_tensors="pt"
                    ).to(device)

                    with torch.no_grad():
                        outputs = model(**inputs)

                    # 6. Post-Process
                    # We only care about the mask inside the Query ROI
                    results = processor.post_process_instance_segmentation(
                        outputs, threshold=0.5, target_sizes=inputs["original_sizes"].tolist()
                    )[0]

                    # 7. Extract Query Prediction
                    pred_final_mask = np.zeros_like(q_gt_mask)
                    q_x1, q_y1, q_x2, q_y2 = query_roi
                    
                    masks = results['masks'].cpu().numpy()
                    
                    for mask in masks:
                        # Crop the mask to the Query Region
                        mask_crop = mask[q_y1:q_y2, q_x1:q_x2]
                        
                        # Resize back to original query size (to match GT)
                        if mask_crop.sum() > 0:
                            mask_orig_size = cv2.resize(
                                mask_crop.astype(np.uint8), 
                                (q_img.width, q_img.height), 
                                interpolation=cv2.INTER_NEAREST
                            )
                            pred_final_mask = np.maximum(pred_final_mask, mask_orig_size)

                    # 8. Compute IoU
                    episode_iou = calculate_iou(pred_final_mask, q_gt_mask)
                    final_results[k][cat_name].append(episode_iou)
                    
                except Exception as e:
                    # In large loops, catch errors so one bad image doesn't kill the script
                    # print(f"Error in episode {episode}: {e}")
                    continue

            # Cleanup VRAM after every class to be safe
            torch.cuda.empty_cache()

    # --- Final Reporting ---
    print("\n\n==========================================")
    print("FINAL THESIS RESULTS")
    print("==========================================")
    
    # Easy Copy-Paste format for LaTeX
    print(f"{'Shot':<5} | {'Class':<20} | {'Mean IoU':<10} | {'Std Dev':<10} | {'95% CI':<10}")
    print("-" * 65)

    for k in shots_list:
        all_class_means = []
        for cat_name in cat_names.values():
            scores = final_results[k][cat_name]
            if not scores:
                continue
                
            mean = statistics.mean(scores) * 100
            stdev = statistics.stdev(scores) * 100 if len(scores) > 1 else 0.0
            # 95% CI = 1.96 * (sigma / sqrt(N))
            ci = 1.96 * (stdev / np.sqrt(len(scores)))
            
            all_class_means.append(mean)
            
            print(f"{k:<5} | {cat_name:<20} | {mean:5.2f}      | {stdev:5.2f}      | ±{ci:4.2f}")
        
        # Calculate mIoU (Global Average)
        if all_class_means:
            global_mean = statistics.mean(all_class_means)
            print(f"{k:<5} | {'*GLOBAL mIoU*':<20} | {global_mean:5.2f}      | --          | --")
        print("-" * 65)

    # Save raw data for plotting later
    with open("sam3_olive_results.json", "w") as f:
        json.dump(final_results, f)
    print("Saved full raw data to sam3_olive_results.json")

if __name__ == "__main__":
    main()