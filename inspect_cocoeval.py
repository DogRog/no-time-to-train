import os
import glob
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
GT_JSON_PATH = "/Users/mac/Developer/VScode/no-time-to-train/data/olive_diseases/annotations/instances_val2017.json"
WORK_DIR = "/Users/mac/Developer/VScode/no-time-to-train/work_dirs/olive_nttt_sam3_eval"

def get_latest_predictions(pattern):
    files = glob.glob(os.path.join(WORK_DIR, "*", pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching {pattern} in {WORK_DIR}")
    return sorted(files, key=os.path.getmtime)[-1]

def get_confusion_matrix_data(coco_eval):
    """
    Extracts matches from COCOeval object for IoU=0.5 (index 0).
    Returns list of dicts with match details.
    """
    if not hasattr(coco_eval, 'evalImgs'):
        print("Please run coco_eval.evaluate() first.")
        return []

    p = coco_eval.params
    # We want IoU=0.5, which is usually at index 0
    # verify if 0.5 is at index 0
    if p.iouThrs[0] != 0.5:
        print(f"Warning: First IoU threshold is {p.iouThrs[0]}, expected 0.5")
    
    iou_idx = 0 
    area_idx = 0 # all areas
    
    matches = []
    
    # evalImgs is a flat list: len(catIds) * len(areaRng) * len(imgIds)
    # The order is: cat -> area -> img
    
    # We need to correctly map the flat index. 
    # Let's rely on the structure:
    # K = len(p.imgIds)
    # A = len(p.areaRng)
    # Index = c * A * K + a * K + i
    
    K = len(p.imgIds)
    A = len(p.areaRng)

    for c_idx, catId in enumerate(p.catIds):
        for i_idx, imgId in enumerate(p.imgIds):
            # Calculate linear index
            entry_idx = c_idx * A * K + area_idx * K + i_idx
            
            entry = coco_eval.evalImgs[entry_idx]
            
            # Entry might be None if no evaluation happened (e.g. no predictions and no GT for that category on that image?)
            # Actually COCOeval usually populates None if ignore constraints are met, 
            # but for standard eval it should be a dict if relevant.
            if entry is None:
                continue
                
            dt_ids = entry['dtIds']
            dt_m = entry['dtMatches'][iou_idx] # Array: for each dt, which gtId it matched (or 0)
            
            # We are interested in:
            # TP: dt matches a gt (dt_m[d] > 0)
            # FP: dt does not match gt (dt_m[d] == 0)
            # FN: gt not matched (we can infer this from gtMatches, or just count GTs - TPs)
            
            # Let's verify with gtMatches to be precise about WHICH gt was missed
            gt_ids = entry['gtIds']
            gt_m = entry['gtMatches'][iou_idx] # Array: for each gt, which dtId matched it (or 0)

            # Record TPs and FPs
            for d_idx, matched_gt_id in enumerate(dt_m):
                if matched_gt_id > 0:
                    matches.append({
                        'image_id': imgId,
                        'category_id': catId,
                        'type': 'TP',
                        'dt_id': dt_ids[d_idx], 
                        'gt_id': matched_gt_id
                    })
                else:
                    matches.append({
                        'image_id': imgId,
                        'category_id': catId,
                        'type': 'FP',
                        'dt_id': dt_ids[d_idx],
                        'gt_id': None
                    })
            
            # Record FNs
            for g_idx, matched_dt_id in enumerate(gt_m):
                if matched_dt_id == 0:
                     matches.append({
                        'image_id': imgId,
                        'category_id': catId,
                        'type': 'FN',
                        'dt_id': None,
                        'gt_id': gt_ids[g_idx]
                    })
    return matches

def compute_confusion_matrix(coco_gt, coco_dt, name="Model"):
    print(f"Evaluating {name}...")
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    match_data = get_confusion_matrix_data(coco_eval)
    df = pd.DataFrame(match_data)
    
    # Map category IDs to names
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_map = {c['id']: c['name'] for c in cats}
    df['category_name'] = df['category_id'].map(cat_map)
    
    # Group by category and type
    cm = df.groupby(['category_name', 'type']).size().unstack(fill_value=0)
    
    # Ensure TP, FP, FN columns exist
    for col in ['TP', 'FP', 'FN']:
        if col not in cm.columns:
            cm[col] = 0
            
    print(f"\nConfusion Matrix for {name}:")
    print(cm)
    
    # Calculate Precision, Recall, F1
    cm['Precision'] = cm['TP'] / (cm['TP'] + cm['FP'])
    cm['Recall'] = cm['TP'] / (cm['TP'] + cm['FN'])
    cm['F1'] = 2 * (cm['Precision'] * cm['Recall']) / (cm['Precision'] + cm['Recall'])
    
    print(f"\nMetrics for {name}:")
    print(cm[['Precision', 'Recall', 'F1']])

    return df

def main():
    try:
        nttt_pred_path = get_latest_predictions("nttt_predictions.json")
        sam3_pred_path = get_latest_predictions("sam3_predictions.json")
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Loading GT from {GT_JSON_PATH}")
    coco_gt = COCO(GT_JSON_PATH)

    print(f"Loading NTTT from {nttt_pred_path}")
    coco_nttt = coco_gt.loadRes(nttt_pred_path)

    print(f"Loading SAM3 from {sam3_pred_path}")
    coco_sam3 = coco_gt.loadRes(sam3_pred_path)

    nttt_matches = compute_confusion_matrix(coco_gt, coco_nttt, "NTTT_Segm")
    sam3_matches = compute_confusion_matrix(coco_gt, coco_sam3, "SAM3_Segm")

if __name__ == "__main__":
    main()


