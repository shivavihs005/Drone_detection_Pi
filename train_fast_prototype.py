import os
import shutil
import glob
from ultralytics import YOLO

def setup_fast_dataset(source_dir, dest_dir, num_images=20):
    images_dir = os.path.join(dest_dir, 'images', 'train')
    labels_dir = os.path.join(dest_dir, 'labels', 'train')
    
    # Create directories
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # We only need 'train' for a fast prototype check
    yaml_content = f"""
path: '{os.path.abspath(dest_dir)}'
train: 'images/train'
val: 'images/train' # Use same for validation to avoid crashing

names:
  0: drone
"""
    with open(os.path.join(dest_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

    # Grab some drone images from the source
    # We prefer images that actually contain drones for the prototype
    search_pattern = os.path.join(source_dir, 'V_DRONE_*.jpg')
    image_files = glob.glob(search_pattern)[:num_images]
    
    if len(image_files) < num_images:
        print(f"Warning: Only found {len(image_files)} drone images (requested {num_images}). We will only use these drone images to ensure accurate training.")

    count = 0
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(source_dir, base_name + '.txt')
        
        if os.path.exists(txt_path):
            shutil.copy(img_path, os.path.join(images_dir, os.path.basename(img_path)))
            shutil.copy(txt_path, os.path.join(labels_dir, os.path.basename(txt_path)))
            count += 1
            
    print(f"Copied {count} images and labels for fast prototype training.")
    return os.path.abspath(os.path.join(dest_dir, 'dataset.yaml'))

if __name__ == '__main__':
    source_test_dir = r"coco json drone detection\test"
    fast_data_dir = "fast_dataset_prototype"
    
    print("Setting up fast 20-image dataset...")
    yaml_path = setup_fast_dataset(source_test_dir, fast_data_dir, num_images=20)
    
    print("\n--- Starting Fast Prototype Training ---")
    # Using yolov8n (nano) which is the smallest standard YOLOv8 model.
    model = YOLO('yolov8n.pt') 
    
    # Train for just 3 epochs on 20 images with a smaller imgsz (320) for speed
    results = model.train(
        data=yaml_path,
        epochs=3,
        imgsz=320,  # 320x320 is much faster for a prototype
        device='cpu',
        batch=4,
        workers=0
    )
    
    print("\n--- Training Complete! ---")
    print(f"Your fast prototype model is saved at: {os.path.abspath('runs/detect/train3/weights/best.pt')} (or runs/detect/trainX/...)")
    print("You can use this .pt file for fast inference.")
