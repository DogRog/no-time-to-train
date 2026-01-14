import os
import json
import argparse

def rename_dataset_files(data_root):
    # Splits to process
    splits = ['train2017', 'val2017', 'test2017']
    
    global_counter = 1
    
    for split in splits:
        print(f"Processing {split}...")
        img_dir = os.path.join(data_root, split)
        ann_file = os.path.join(data_root, 'annotations', f'instances_{split}.json')
        
        if not os.path.exists(img_dir):
            print(f"Directory {img_dir} does not exist. Skipping.")
            continue
            
        if not os.path.exists(ann_file):
        
            print(f"Annotation file {ann_file} does not exist. Skipping.")
            continue
            
        # Get list of files, ignore hidden files
        files = [f for f in os.listdir(img_dir) if not f.startswith('.')]
        # Sort to ensure deterministic order
        files.sort()
        
        filename_mapping = {}
        
        # Rename files in the directory
        for old_name in files:
            old_path = os.path.join(img_dir, old_name)
            if os.path.isdir(old_path):
                continue
                
            # Rename files to 12-digit zero-padded format
            new_name = f"{global_counter:012d}.jpg"
            new_path = os.path.join(img_dir, new_name)
            
            # Avoid overwriting if source and dest are same (already renamed)
            if old_name == new_name:
                filename_mapping[old_name] = new_name
                global_counter += 1
                continue
            
             # Check collision
            if os.path.exists(new_path):
                 print(f"Warning: Destination {new_path} exists. Overwriting.")
                
            os.rename(old_path, new_path)
            filename_mapping[old_name] = new_name
            global_counter += 1
            
        # Update annotations JSON
        with open(ann_file, 'r') as f:
            data = json.load(f)

        # Remove diseases-usdB category
        if 'categories' in data:
            # Find the ID of diseases-usdB
            cat_to_remove = None
            for cat in data['categories']:
                if cat['name'] == 'diseases-usdB':
                    cat_to_remove = cat
                    break
            
            if cat_to_remove:
                cat_id = cat_to_remove['id']
                print(f"Removing category: {cat_to_remove['name']} (ID: {cat_id})")
                
                # Remove the category
                data['categories'] = [c for c in data['categories'] if c['id'] != cat_id]
                
                # Remove annotations with this category ID
                if 'annotations' in data:
                    original_ann_count = len(data['annotations'])
                    data['annotations'] = [ann for ann in data['annotations'] if ann['category_id'] != cat_id]
                    removed_ann_count = original_ann_count - len(data['annotations'])
                    if removed_ann_count > 0:
                        print(f"Removed {removed_ann_count} annotations for category {cat_to_remove['name']}")
                
                # Update supercategory for other categories
                for cat in data['categories']:
                    if cat.get('supercategory') == 'diseases-usdB':
                        cat['supercategory'] = 'none'
            
        updated_count = 0
        if 'images' in data:
            for img in data['images']:
                original_fname = img.get('file_name')
                if original_fname and original_fname in filename_mapping:
                    img['file_name'] = filename_mapping[original_fname]
                    updated_count += 1
                # If the image name in json was already renamed (or different reason), we might skip or not find it.
                # But here we map based on what was on disk.
                # If file on disk matches mapped name, we are good.
        
        # Update `info` to match COCO standard structure if possible
        if 'info' in data:
            # Reorder keys: description -> url -> version -> year -> contributor -> date_created
            # File B (COCO): description -> url -> version -> year -> contributor -> date_created
            info_data = data['info']
            new_info = {
                "description": info_data.get("description", "Olive Disease Dataset"),
                "url": info_data.get("url", ""),
                "version": info_data.get("version", "1.0"),
                "year": 2017, # Set to 2017 to match the folder structure / COCO year
                "contributor": info_data.get("contributor", ""),
                "date_created": info_data.get("date_created", "")
            }
            data['info'] = new_info

        print(f"Updated {updated_count} images in {ann_file}")
        
        # Minify JSON by removing whitespace separators to match typical COCO format
        with open(ann_file, 'w') as f:
            json.dump(data, f, separators=(',', ':'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rename olive dataset images and update annotations.")
    parser.add_argument('--data_root', type=str, required=True, help="Root directory of the dataset (e.g., data/olive_diseases)")
    
    args = parser.parse_args()
    
    rename_dataset_files(args.data_root)
