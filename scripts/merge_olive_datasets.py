import json
import os
import shutil
from pathlib import Path

def merge_olive_datasets():
    base_dir = Path("data/olive_diseases")
    splits = ["train2017", "val2017", "test2017"]
    
    merged_json = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    # Create all_images directory
    all_images_dir = base_dir / "all_images"
    all_images_dir.mkdir(exist_ok=True)
    
    img_id_offset = 0
    ann_id_offset = 0
    
    cat_map = {} # old_cat_id -> new_cat_id (should be same, but to be safe)
    
    # Global counters for unique IDs across all splits
    global_img_id = 0
    global_ann_id = 0

    # Load categories from the first available split
    first_file = base_dir / "annotations" / f"instances_{splits[0]}.json"
    if first_file.exists():
        with open(first_file, "r") as f:
            data = json.load(f)
            merged_json["categories"] = data["categories"]
            merged_json["info"] = data.get("info", {})
            merged_json["licenses"] = data.get("licenses", [])
        print(f"Loaded categories from {first_file}")
    else:
        print(f"Error: {first_file} not found.")
        return

    for split in splits:
        json_file = base_dir / "annotations" / f"instances_{split}.json"
        img_dir = base_dir / split
        
        if not json_file.exists():
            print(f"Skipping {split}, JSON not found.")
            continue
            
        print(f"Processing {split}...")
        with open(json_file, "r") as f:
            data = json.load(f)
            
        # Map old image IDs to new global IDs for this split
        split_img_id_map = {}
        
        # Process Images
        for img in data["images"]:
            old_img_id = img["id"]
            
            # Create new unique ID
            new_img_id = global_img_id
            global_img_id += 1
            
            split_img_id_map[old_img_id] = new_img_id
            
            # Symlink image to all_images
            src_img = img_dir / img["file_name"]
            dst_img = all_images_dir / img["file_name"]
            
            if not dst_img.exists() and src_img.exists():
                os.symlink(src_img.resolve(), dst_img)
            
            # Add to merged with new ID
            new_img = img.copy()
            new_img["id"] = new_img_id
            merged_json["images"].append(new_img)
               
        # Process Annotations
        for ann in data["annotations"]:
            old_img_id = ann["image_id"]
            
            # Only add annotation if we have the image
            if old_img_id in split_img_id_map:
                new_ann = ann.copy()
                new_ann["id"] = global_ann_id
                global_ann_id += 1
                new_ann["image_id"] = split_img_id_map[old_img_id]
                
                merged_json["annotations"].append(new_ann)
            else:
                 print(f"Warning: Annotation {ann['id']} references unknown image {old_img_id} in {split}")

    out_file = base_dir / "annotations" / "instances_all.json"
    with open(out_file, "w") as f:
        json.dump(merged_json, f)
    
    print(f"Merged dataset saved to {out_file}")
    print(f"Total images: {len(merged_json['images'])}")
    print(f"Total annotations: {len(merged_json['annotations'])}")

if __name__ == "__main__":
    merge_olive_datasets()
