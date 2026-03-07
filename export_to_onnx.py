"""
Export YOLO .pt model to .onnx format for Raspberry Pi.
Run this on Windows/Mac where PyTorch works, NOT on the Pi.

Usage:
    python export_to_onnx.py

This will export the best available .pt model to models/yolov8n.onnx
Then copy models/yolov8n.onnx to the Pi.
"""
import os
from ultralytics import YOLO


def export():
    # Find the best model (same logic as camera_module.py)
    custom_model = 'runs/detect/train4/weights/best.pt'
    fallback1 = 'models/yolov8n.pt'
    fallback2 = 'yolov8n.pt'

    if os.path.exists(custom_model):
        model_path = custom_model
    elif os.path.exists(fallback1):
        model_path = fallback1
    else:
        model_path = fallback2

    print(f"[EXPORT] Loading model: {model_path}")
    model = YOLO(model_path)

    # Export to ONNX
    os.makedirs('models', exist_ok=True)
    print("[EXPORT] Exporting to ONNX format...")
    model.export(format='onnx', imgsz=640, simplify=True)

    # The export creates the .onnx file next to the .pt file
    onnx_path = model_path.replace('.pt', '.onnx')
    target_path = 'models/yolov8n.onnx'

    if os.path.exists(onnx_path) and onnx_path != target_path:
        import shutil
        shutil.copy2(onnx_path, target_path)
        print(f"[EXPORT] Copied to {target_path}")

    # Also save the class names so the Pi code knows the labels
    names = model.names  # dict like {0: 'person', 1: 'bicycle', ...}
    names_path = 'models/class_names.txt'
    with open(names_path, 'w') as f:
        for idx in sorted(names.keys()):
            f.write(f"{names[idx]}\n")
    print(f"[EXPORT] Class names saved to {names_path}")

    print(f"\n[EXPORT] Done! Copy these files to the Pi:")
    print(f"  - {target_path}")
    print(f"  - {names_path}")


if __name__ == '__main__':
    export()
