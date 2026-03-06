import threading
import time
import cv2
import numpy as np
import os
from ultralytics import YOLO

class CameraModule:
    def __init__(self, model_path='models/yolov8n.pt'):
        self._thread = None
        self._running = False
        self._current_frame = None
        self._current_confidence = 0.0
        self._lock = threading.Lock()
        self.vision_enabled = True
        
        # Proxy labels: 'drone' for our custom model.
        # When falling back to base YOLOv8n (COCO), also watch for 'bird' / 'airplane'
        # as rough stand-ins so the video feed is still annotated.
        self.proxy_labels = {"drone", "bird", "airplane"}

        # Load YOLO model
        print("[CAMERA] Loading YOLO model...")
        
        # YOLOv8n automatically downloads to the current dir if not present
        if not os.path.exists('models'):
            os.makedirs('models')

        # --- Model path resolution (3-tier fallback) ---
        # 1. Best custom-trained model from training run
        # 2. Pre-downloaded YOLOv8n inside models/
        # 3. Root-level yolov8n.pt (auto-downloaded by ultralytics if missing)
        custom_model  = 'runs/detect/train4/weights/best.pt'
        fallback1     = 'models/yolov8n.pt'
        fallback2     = 'yolov8n.pt'

        if os.path.exists(custom_model):
            model_path = custom_model
            print(f"[CAMERA] Using custom-trained model: {model_path}")
        elif os.path.exists(fallback1):
            model_path = fallback1
            print(f"[CAMERA] Custom model not found. Using fallback: {model_path}")
        else:
            model_path = fallback2   # ultralytics will auto-download if absent
            print(f"[CAMERA] Using base YOLOv8n model: {model_path} (will auto-download if needed)")

        self.model = YOLO(model_path)
        
        # Try to initialize Picamera2
        try:
            from picamera2 import Picamera2
            print("[CAMERA] Initializing Picamera2...")
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration({"size": (1280, 720)})
            self.picam2.configure(config)
            self.picam2.start()
            self.use_picam = True
            print("[CAMERA] Picamera2 started.")
        except ImportError:
            print("[CAMERA] Picamera2 not found. Falling back to OpenCV video capture (for local testing).")
            self.use_picam = False
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.use_picam:
            self.picam2.stop()
        else:
            self.cap.release()

    def _capture_loop(self):
        # We target ~15 FPS. Inference usually takes some time, so loop delay can be minimal.
        while self._running:
            if self.use_picam:
                # Picamera2 capture
                try:
                    frame = self.picam2.capture_array()
                    # YOLO expects 3 channels. Picamera2 may return 4 channels (e.g. RGBA/XBGR8888).
                    if len(frame.shape) == 3 and frame.shape[2] == 4:
                        frame = np.ascontiguousarray(frame[:, :, :3])
                except Exception as e:
                    print(f"[CAMERA] Capture error: {e}")
                    time.sleep(0.1)
                    continue
            else:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

            highest_conf = 0.0
            best_box = None
            label_text = ""

            if self.vision_enabled:
                # Run YOLO inference
                results = self.model(frame, verbose=False)
                result = results[0]
                names = result.names
                
                for box in result.boxes:
                    cls_idx = int(box.cls[0].item())
                    label = names.get(cls_idx, str(cls_idx)).lower()
                    
                    if label not in self.proxy_labels:
                        continue # Skip people, cars, chairs, etc.
                        
                    conf = float(box.conf[0])
                    if conf > highest_conf:
                        highest_conf = conf
                        best_box = list(map(int, box.xyxy[0]))
                        label_text = f"possible_drone({label}): {highest_conf:.2f}"

            # Draw bounding box if we found something
            if best_box and self.vision_enabled:
                x1, y1, x2, y2 = best_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif not self.vision_enabled:
                cv2.putText(frame, "VISION SENSOR DISABLED", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            
            with self._lock:
                if ret:
                    self._current_frame = buffer.tobytes()
                # Requirements: If no detection -> confidence = 0
                self._current_confidence = highest_conf
                
            # Yield briefly to not peg CPU 100%
            time.sleep(0.01)

    def get_latest(self):
        with self._lock:
            return {
                "frame": self._current_frame,
                "confidence": self._current_confidence,
                "vision_enabled": self.vision_enabled
            }
