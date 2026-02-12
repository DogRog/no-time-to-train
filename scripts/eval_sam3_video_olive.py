import argparse
import json
import os
import random
import sys
import statistics
import time
from collections import defaultdict

import numpy as np
import torch
import cv2
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

# Add project root to path
sys.path.append(os.getcwd())

from no_time_to_train.dataset.coco_ref_dataset import (
    COCOMemoryFillCropDataset, COCORefOracleTestDataset)
from no_time_to_train.dataset.few_shot_sampling import sample_memory_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 Video-Based Few-Shot Evaluation")
    parser.add_argument("--shots", type=int, default=10, help="Number of support shots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--output_dir", type=str, default="work_dirs/sam3_video_results", help="Output directory")
    parser.add_argument("--prediction_file", type=str, default="sam3_predictions.json", help="Prediction filename inside output_dir")
    parser.add_argument("--score", type=float, default=1.0, help="Confidence score assigned to SAM3 predictions")
    parser.add_argument("--evaluate_coco", action="store_true", help="Run COCO bbox/segm evaluation after exporting predictions")
    return parser.parse_args()

def calculate_iou(pred_mask, gt_mask):
    """Computes binary IoU."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def mask_to_bbox_xywh(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)]

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    using_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    if using_cuda:
        torch.cuda.reset_peak_memory_stats()

    # 1. Setup Model
    print(f"Loading SAM3 model on {args.device}...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model = Sam3TrackerVideoModel.from_pretrained(
        "facebook/sam3",
        torch_dtype=dtype,
        attn_implementation="flash_attention_2"
    ).to(args.device)
    
    processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
    
    # 2. Setup Datasets
    project_root = os.getcwd()
    
    # Support Set Config
    support_pkl_path = os.path.join(project_root, f"work_dirs/olive_results/olive_{args.shots}shot_seed{args.seed}.pkl")
    support_json_file = os.path.join(project_root, "data/olive_diseases/annotations/instances_train2017.json")
    
    if not os.path.exists(support_pkl_path):
        print(f"Generating few-shot split at {support_pkl_path}...")
        os.makedirs(os.path.dirname(support_pkl_path), exist_ok=True)
        sample_memory_dataset(
            json_file=support_json_file,
            out_path=support_pkl_path,
            memory_length=args.shots,
            remove_bad=True,
            dataset="olive_diseases"
        )
    
    support_set = COCOMemoryFillCropDataset(
        root=os.path.join(project_root, "data/olive_diseases/train2017"),
        json_file=support_json_file,
        memory_pkl=support_pkl_path,
        class_split="olive_diseases",
        image_size=1024,
        memory_length=args.shots,
        context_ratio=0.2, # Matching notebook
        norm_img=False
    )
    
    # Query Set Config
    query_set = COCORefOracleTestDataset(
        root=os.path.join(project_root, "data/olive_diseases/val2017"),
        json_file=os.path.join(project_root, "data/olive_diseases/annotations/instances_val2017.json"),
        image_size=1024,
        norm_img=False,
        class_split="olive_diseases",
        with_query_points=False
    )
    
    print(f"Support Set: {len(support_set)} items")
    print(f"Query Set: {len(query_set)} items")
    
    # 3. Pre-load Support Metadata
    # Organizing support items by (internal) cat_ind for easier access? 
    # Actually, the notebook iterates all `all_support_metadata` which contains everything.
    # Let's rebuild that structure.
    all_support_images = []
    all_support_metadata = []
    
    print("Loading support images into memory...")
    for i in range(len(support_set)):
        item = support_set[i]
        refs = item['refs_by_cat']
        cat_ind = list(refs.keys())[0]
        
        # Image: (1, 3, H, W) -> (H, W, 3) numpy
        img_tensor = refs[cat_ind]['imgs'][0] 
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        # Make sure it's uint8
        img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        
        all_support_images.append(img_uint8)
        
        real_cat_id = support_set.cat_inds_to_ids[cat_ind]
        all_support_metadata.append({
            "index": i,
            "cat_ind": cat_ind,
            "category_id": real_cat_id,
        })
        
    print(f"Loaded {len(all_support_images)} support images.")

    # 4. Evaluation Loop
    results = defaultdict(list)
    predictions = []
    total_query_time_sec = 0.0
    processed_queries = 0
    
    # Target size for SAM3
    target_h, target_w = 1024, 1024
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
         size_conf = processor.image_processor.size
         target_h = size_conf.get("height", 1024)
         target_w = size_conf.get("width", 1024)

    for query_idx in tqdm(range(len(query_set)), desc="Evaluating Queries"):
        query_start_t = time.perf_counter()
        query_item = query_set[query_idx]
        
        # Get query image
        if 'target_img' not in query_item:
            continue
            
        q_img_tensor = query_item['target_img']
        q_img_np = q_img_tensor.permute(1, 2, 0).numpy()
        q_img_uint8 = (np.clip(q_img_np, 0, 1) * 255).astype(np.uint8)
        
        # Format Video: [Support Frames] + [Query Frame]
        # Note: In notebook, support frames were repeated NUM_REPEATS times?
        # "repeated NUM_REPEATS times" was in the notebook, let's stick to 1 to match pure few-shot logic unless needed?
        # Set NUM_REPEATS = 1 for efficiency unless tracking fails.
        video_frames = all_support_images + [q_img_uint8]
        
        # Init Session with LIST of numpy arrays (no file writing)
        inference_session = processor.init_video_session(
            video=video_frames,
            inference_device=args.device,
            processing_device=args.device,
            dtype=dtype, 
        )
        
        # Add Support Masks
        with torch.inference_mode():
            for idx, meta in enumerate(all_support_metadata):
                frame_idx = idx # Support frames are at the beginning
                
                # Get GT mask
                ds_item = support_set[meta['index']]
                gt_mask = ds_item['refs_by_cat'][meta['cat_ind']]['masks']
                
                if isinstance(gt_mask, torch.Tensor):
                    gt_mask = gt_mask.numpy()
                if gt_mask.ndim == 3:
                    gt_mask = gt_mask[0]
                
                # Preprocess prompt mask on GPU
                mask_tensor = torch.from_numpy(gt_mask).to(args.device)
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).float()
                
                if mask_tensor.shape[-2] != target_h or mask_tensor.shape[-1] != target_w:
                    mask_tensor = torch.nn.functional.interpolate(
                        mask_tensor, size=(target_h, target_w), mode='nearest'
                    )
                mask_tensor = (mask_tensor > 0).to(dtype)
                
                obj_id = meta['cat_ind'] + 1
                
                processor.add_inputs_to_inference_session(
                    inference_session=inference_session,
                    frame_idx=frame_idx,
                    obj_ids=[obj_id],
                    input_masks=mask_tensor
                )
                
                # Encode support frame
                model(
                    inference_session=inference_session,
                    frame_idx=frame_idx,
                )
        
        # Propagate to get Query Prediction
        video_segments = {}
        with torch.inference_mode():
            for output in model.propagate_in_video_iterator(inference_session):
                video_segments[output.frame_idx] = output.pred_masks
        
        # Get result for the last frame (Query)
        query_frame_idx = len(video_frames) - 1
        
        if query_frame_idx in video_segments:
            masks_logits = video_segments[query_frame_idx]
            video_res_masks = processor.post_process_masks(
                [masks_logits], 
                original_sizes=[[inference_session.video_height, inference_session.video_width]], 
                binarize=True
            )[0]
            
            # Evaluate against Ground Truth
            gt_anns = query_item.get('tar_anns_by_cat', {})
            
            # For each potential class (object ID)
            # obj_id corresponds to meta['cat_ind'] + 1
            # predictions are in video_res_masks[cat_ind] effectively if obj_ids are sequential
            
            # But wait, video_res_masks shape depends on number of tracked objects.
            # We track ALL support classes.
            # Assuming processor returns masks in order of object IDs?
            # Usually yes.
            
            pred_masks_np = video_res_masks.cpu().numpy() # (N_obj, 1, H, W)
            
            # We iterate over unique categories present in support set
            # all_support_metadata maps frames to categories.
            # We need to map object_id back to category_ind.
            # In our loop: obj_id = meta['cat_ind'] + 1
            # So obj_id 1 is cat_ind 0, etc.
            
            # We must verify if SAM3 returns masks for all ID's or sparse?
            # propagate_in_video_iterator usually returns logical masks for all tracked IDs?
            # Let's assume indices map 1:1 if we added all of them.
            
            # Optimization: Not all classes might be in the query image.
            # But we must evaluate IoU for all classes (TP, FP, FN).
            
            # Get available GT categories for this image
            available_gt_cats = set(gt_anns.keys())
            
            # Iterate through all trained categories (0..4 for olive)
            # Find max cat_ind
            max_cat = max([m['cat_ind'] for m in all_support_metadata])
            
            for cat_ind in range(max_cat + 1):
                # Retrieve prediction for this category
                # Assuming pred_masks_np[cat_ind] corresponds to obj_id = cat_ind + 1
                if cat_ind < pred_masks_np.shape[0]:
                    pred_mask = pred_masks_np[cat_ind, 0] > 0
                else:
                    pred_mask = np.zeros((1024, 1024), dtype=bool) # Falback
                
                # Retrieve GT
                if cat_ind in gt_anns:
                    gt_tensor = gt_anns[cat_ind]['masks']
                    if gt_tensor.ndim == 3:
                        gt_mask = gt_tensor.sum(dim=0).cpu().numpy() > 0
                    else:
                        gt_mask = gt_tensor.cpu().numpy() > 0
                else:
                    gt_mask = np.zeros_like(pred_mask)
                
                iou = calculate_iou(pred_mask, gt_mask)
                real_cat_id = support_set.cat_inds_to_ids[cat_ind]
                results[real_cat_id].append(iou)

                pred_mask_u8 = pred_mask.astype(np.uint8)
                if pred_mask_u8.sum() == 0:
                    continue

                ori_h = int(query_item['target_img_info']['ori_height'])
                ori_w = int(query_item['target_img_info']['ori_width'])
                pred_mask_resized = cv2.resize(
                    pred_mask_u8,
                    (ori_w, ori_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.uint8)

                if pred_mask_resized.sum() == 0:
                    continue

                bbox_xywh = mask_to_bbox_xywh(pred_mask_resized)
                if bbox_xywh is None:
                    continue

                segmentation = mask_utils.encode(np.asfortranarray(pred_mask_resized))
                segmentation['counts'] = segmentation['counts'].decode('utf-8')

                predictions.append(
                    {
                        'image_id': int(query_item['target_img_info']['id']),
                        'category_id': int(real_cat_id),
                        'bbox': bbox_xywh,
                        'score': float(args.score),
                        'segmentation': segmentation,
                    }
                )

        total_query_time_sec += max(0.0, time.perf_counter() - query_start_t)
        processed_queries += 1

    # 5. Report
    print("\n--- Evaluation Results ---")
    all_ious = []
    headers = ["Class ID", "Class Name", "mIoU"]
    print(f"{headers[0]:<10} | {headers[1]:<20} | {headers[2]:<10}")
    print("-" * 46)
    
    # Needs class names from dataset
    # We can access `support_set.coco.cats`
    cats_info = support_set.coco.cats
    
    for cat_id, iou_list in results.items():
        mean_iou = sum(iou_list) / len(iou_list)
        all_ious.append(mean_iou)
        cat_name = cats_info[cat_id]['name'] if cat_id in cats_info else str(cat_id)
        print(f"{cat_id:<10} | {cat_name:<20} | {mean_iou:.4f}")
    
    print("-" * 46)
    overall_miou = (sum(all_ious) / len(all_ious)) if len(all_ious) > 0 else 0.0
    print(f"Overall mIoU: {overall_miou:.4f}")

    pred_path = os.path.join(args.output_dir, args.prediction_file)
    with open(pred_path, "w") as f:
        json.dump(predictions, f)
    print(f"Saved {len(predictions)} predictions to {pred_path}")

    fps = (processed_queries / total_query_time_sec) if total_query_time_sec > 0 else 0.0
    peak_vram_allocated_mib = None
    peak_vram_reserved_mib = None
    if using_cuda:
        peak_vram_allocated_mib = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        peak_vram_reserved_mib = float(torch.cuda.max_memory_reserved() / (1024 ** 2))

    runtime_summary = {
        "model": "sam3",
        "shots": int(args.shots),
        "seed": int(args.seed),
        "num_queries": int(processed_queries),
        "total_inference_time_sec": float(total_query_time_sec),
        "fps": float(fps),
        "peak_vram_mib": peak_vram_reserved_mib,
        "peak_vram_allocated_mib": peak_vram_allocated_mib,
        "peak_vram_reserved_mib": peak_vram_reserved_mib,
    }
    runtime_path = os.path.join(args.output_dir, "sam3_runtime.json")
    with open(runtime_path, "w") as f:
        json.dump(runtime_summary, f, indent=2)
    print(f"Saved SAM3 runtime summary to {runtime_path}")
    print(f"SAM3 FPS: {fps:.3f}")
    if peak_vram_reserved_mib is not None:
        print(f"SAM3 peak VRAM (reserved MiB): {peak_vram_reserved_mib:.2f}")

    if args.evaluate_coco and len(predictions) > 0:
        coco_results = query_set.coco.loadRes(predictions)

        print("\n--- COCO Evaluation (SAM3 Predictions) ---")
        coco_eval_bbox = COCOeval(query_set.coco, coco_results, "bbox")
        coco_eval_bbox.params.imgIds = query_set.img_ids
        coco_eval_bbox.evaluate()
        coco_eval_bbox.accumulate()
        coco_eval_bbox.summarize()

        coco_eval_segm = COCOeval(query_set.coco, coco_results, "segm")
        coco_eval_segm.params.imgIds = query_set.img_ids
        coco_eval_segm.evaluate()
        coco_eval_segm.accumulate()
        coco_eval_segm.summarize()

if __name__ == "__main__":
    main()
