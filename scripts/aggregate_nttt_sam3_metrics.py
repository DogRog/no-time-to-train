import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


RUN_DIR_PATTERN = re.compile(r"^(?P<version>.+)_(?P<shots>\d+)shot_seed(?P<seed>\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate NTTT/SAM3 COCO metrics and FPS across run folders into one CSV."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="work_dirs/olive_nttt_sam3_eval",
        help="Root directory containing run folders like <version>_<shots>shot_seed<seed>",
    )
    parser.add_argument(
        "--gt_json",
        type=str,
        default="data/olive_diseases/annotations/instances_val2017.json",
        help="Ground-truth COCO annotation JSON used for evaluation",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="work_dirs/olive_nttt_sam3_eval/metrics_summary.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def run_coco_eval(coco_gt, predictions, iou_type="segm"):
    if len(predictions) == 0:
        raise ValueError("Prediction list is empty")

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.params.imgIds = sorted(coco_gt.getImgIds())
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR@1": float(stats[6]),
        "AR@10": float(stats[7]),
        "AR@100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
    }


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_runtime_stat(runtime_path, key):
    if not runtime_path.exists():
        return math.nan
    data = load_json(runtime_path)
    value = data.get(key, math.nan)
    return float(value) if value is not None else math.nan


def collect_rows_for_model(coco_gt, run_dir, version, shots, seed, model_name, pred_file, runtime_file):
    pred_path = run_dir / pred_file
    runtime_path = run_dir / runtime_file

    if not pred_path.exists():
        return None

    predictions = load_json(pred_path)
    if len(predictions) == 0:
        return {
            "run_dir": run_dir.name,
            "version": version,
            "shots": int(shots),
            "seed": int(seed),
            "model": model_name,
            "fps": load_runtime_stat(runtime_path, "fps"),
            "peak_vram_mib": load_runtime_stat(runtime_path, "peak_vram_mib"),
            "num_predictions": 0,
        }

    bbox_stats = run_coco_eval(coco_gt, predictions, iou_type="bbox")
    segm_stats = run_coco_eval(coco_gt, predictions, iou_type="segm")

    row = {
        "run_dir": run_dir.name,
        "version": version,
        "shots": int(shots),
        "seed": int(seed),
        "model": model_name,
        "fps": load_runtime_stat(runtime_path, "fps"),
        "peak_vram_mib": load_runtime_stat(runtime_path, "peak_vram_mib"),
        "num_predictions": len(predictions),
    }
    row.update({f"bbox_{k}": v for k, v in bbox_stats.items()})
    row.update({f"segm_{k}": v for k, v in segm_stats.items()})
    return row


def main():
    args = parse_args()

    root_dir = Path(args.root_dir)
    out_csv = Path(args.out_csv)
    gt_json = Path(args.gt_json)

    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir not found: {root_dir}")
    if not gt_json.exists():
        raise FileNotFoundError(f"gt_json not found: {gt_json}")

    coco_gt = COCO(str(gt_json))

    rows = []
    run_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    for run_dir in run_dirs:
        match = RUN_DIR_PATTERN.match(run_dir.name)
        if match is None:
            continue

        version = match.group("version")
        shots = match.group("shots")
        seed = match.group("seed")

        nttt_row = collect_rows_for_model(
            coco_gt,
            run_dir,
            version,
            shots,
            seed,
            model_name="NTTT",
            pred_file="nttt_predictions.json",
            runtime_file="nttt_runtime.json",
        )
        if nttt_row is not None:
            rows.append(nttt_row)

        sam3_row = collect_rows_for_model(
            coco_gt,
            run_dir,
            version,
            shots,
            seed,
            model_name="SAM3",
            pred_file="sam3_predictions.json",
            runtime_file="sam3_runtime.json",
        )
        if sam3_row is not None:
            rows.append(sam3_row)

    if len(rows) == 0:
        raise RuntimeError(
            "No valid runs found. Ensure run folders follow '<version>_<shots>shot_seed<seed>' and contain prediction JSON files."
        )

    df = pd.DataFrame(rows)
    sort_cols = ["version", "shots", "seed", "model"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Saved summary CSV: {out_csv}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
