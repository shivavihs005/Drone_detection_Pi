import json
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

def convert_coco_to_yolo(split_path):
    json_path = os.path.join(split_path, '_annotations.coco.json')
    if not os.path.exists(json_path):
        print(f"Skipping {split_path}: no _annotations.coco.json found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # create a category mapping
    # ensure everything merges into a single class 'drone' (id 0) or map correctly
    # Roboflow sometimes exports 0 for background. We map everything to '0' = drone
    
    img_dict = {img['id']: img for img in data['images']}
    
    for ann in data['annotations']:
        img = img_dict[ann['image_id']]
        img_w = img['width']
        img_h = img['height']
        
        # COCO bbox is [top_left_x, top_left_y, width, height]
        x_tl, y_tl, w, h = ann['bbox']
        
        x_center = x_tl + (w / 2.0)
        y_center = y_tl + (h / 2.0)
        
        # Normalize
        x_center /= img_w
        y_center /= img_h
        w /= img_w
        h /= img_h
        
        # Write to txt file
        txt_filename = os.path.splitext(img['file_name'])[0] + '.txt'
        txt_path = os.path.join(split_path, txt_filename)
        
        # Class 0: drone
        with open(txt_path, 'a') as text_file:
            text_file.write(f"0 {x_center} {y_center} {w} {h}\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLOv8 on a COCO dataset")
    parser.add_argument('--dataset', type=str, default='coco json drone detection', help="Path to the COCO dataset directory")
    args = parser.parse_args()
    
    dataset_dir = args.dataset
    dataset_dir_abs = os.path.abspath(dataset_dir)
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        print("Please ensure the folder exists or provide the correct path using: python train_custom_drone.py --dataset <path>")
        exit(1)
    
    print(f"Using dataset directory: {dataset_dir_abs}")
    print("Converting COCO labels to YOLO format...")
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(dataset_dir_abs, split)
        if os.path.exists(split_path):
            convert_coco_to_yolo(split_path)
            
    print("Creating dataset.yaml...")
    yaml_content = f"""
path: '{dataset_dir_abs}'
train: 'train'
val: 'valid'
test: 'test'

names:
  0: drone
"""
    yaml_path = os.path.join(dataset_dir_abs, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print("Formatting complete! Starting YOLO training now...")
    model = YOLO('yolov8n.pt')  # Start from pretrained nano model
    model.train(data=yaml_path, epochs=10, imgsz=640, device='cpu')  # Auto-detects device, falls back to CPU
    print("Training finished! You can find your new model in 'runs/detect/train/weights/best.pt'")
