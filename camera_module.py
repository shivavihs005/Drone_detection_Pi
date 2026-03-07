import threading
import time
import cv2
import os
import numpy as np

class CameraModule:
    def __init__(self, model_path='models/yolov8n.onnx'):
        self._thread = None
        self._running = False
        self._current_frame = None
        self._current_confidence = 0.0
        self._vision_detected = False
        self._lock = threading.Lock()
        self.vision_enabled = True
        self.VISION_THRESHOLD = 0.4  # confidence threshold for vision-only drone detection
        self.CONF_THRESHOLD = 0.25
        self.INPUT_SIZE = 640
        
        # Proxy labels: 'drone' for our custom model.
        # When falling back to base YOLOv8n (COCO), also watch for 'bird' / 'airplane'
        # as rough stand-ins so the video feed is still annotated.
        self.proxy_labels = {"drone", "bird", "airplane"}

        self.class_names = self._load_class_names()

        # OpenCV DNN backend state
        self._backend = "none"
        self.model = None

        # Load ONNX model with OpenCV DNN (Pi-safe: no torch/ultralytics import needed)
        print("[CAMERA] Loading vision model (OpenCV DNN ONNX)...")
        os.makedirs('models', exist_ok=True)
        model_path = self._resolve_model_path(model_path)
        if model_path:
            try:
                self.model = cv2.dnn.readNetFromONNX(model_path)
                self._backend = "opencv_onnx"
                print(f"[CAMERA] Using ONNX model: {model_path}")
            except Exception as e:
                print(f"[CAMERA] Failed to load ONNX model: {e}")
                self._backend = "none"
        else:
            print("[CAMERA] No ONNX model found. Vision confidence will stay 0.0.")
            print("[CAMERA] Generate model on dev machine: python export_to_onnx.py")
        
        # Use USB webcam via OpenCV
        print("[CAMERA] Initializing USB webcam via OpenCV...")
        self.cap = self._open_camera()
        if self.cap is None:
            raise RuntimeError("[CAMERA] Could not open USB webcam. Check connection and /dev/video*.")
        print("[CAMERA] USB webcam started.")

    def _open_camera(self):
        # Prefer V4L2 backend on Raspberry Pi to avoid GStreamer pipeline issues.
        device_paths = ["/dev/video0", "/dev/video1"]
        for path in device_paths:
            if os.path.exists(path):
                cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
                if cap is not None and cap.isOpened():
                    # USB UVC cams on Pi are often more stable with MJPG.
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)

                    # Accept only if the device returns an actual frame.
                    ok, _ = cap.read()
                    if ok:
                        print(f"[CAMERA] Opened camera device {path} (V4L2).")
                        return cap
                    cap.release()

        candidates = [
            (0, cv2.CAP_V4L2),
            (1, cv2.CAP_V4L2),
            (0, cv2.CAP_ANY),
            (1, cv2.CAP_ANY),
        ]
        for idx, backend in candidates:
            cap = cv2.VideoCapture(idx, backend)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                ok, _ = cap.read()
                if ok:
                    print(f"[CAMERA] Opened camera index {idx} backend {backend}.")
                    return cap
                cap.release()
            if cap is not None:
                cap.release()
        return None

    def _resolve_model_path(self, requested_path):
        candidates = [
            'runs/detect/train4/weights/best.onnx',
            'models/yolov8n.onnx',
            requested_path,
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None

    def _load_class_names(self):
        names_file = 'models/class_names.txt'
        if os.path.exists(names_file):
            names = {}
            with open(names_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    names[idx] = line.strip().lower()
            return names
        coco = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        return {i: name for i, name in enumerate(coco)}

    def _infer_opencv_onnx(self, frame):
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(self.INPUT_SIZE, self.INPUT_SIZE),
            swapRB=True,
            crop=False,
        )
        self.model.setInput(blob)
        output = self.model.forward()

        if output.ndim == 3:
            output = output[0]
        if output.shape[0] < output.shape[1]:
            output = output.T  # (84, N) -> (N, 84)

        frame_h, frame_w = frame.shape[:2]
        x_factor = frame_w / float(self.INPUT_SIZE)
        y_factor = frame_h / float(self.INPUT_SIZE)

        boxes = []
        confidences = []
        class_ids = []
        num_classes = max(output.shape[1] - 4, 0)

        for row in output:
            if num_classes <= 0:
                continue
            cls_scores = row[4:]
            cls_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[cls_id])
            if conf < self.CONF_THRESHOLD:
                continue

            cx, cy, w, h = row[:4]
            x1 = int((cx - w / 2) * x_factor)
            y1 = int((cy - h / 2) * y_factor)
            bw = int(w * x_factor)
            bh = int(h * y_factor)

            boxes.append([x1, y1, bw, bh])
            confidences.append(conf)
            class_ids.append(cls_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONF_THRESHOLD, 0.45)
        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                x1, y1, bw, bh = boxes[idx]
                x2 = x1 + bw
                y2 = y1 + bh
                label = self.class_names.get(class_ids[idx], str(class_ids[idx]))
                detections.append(
                    {
                        "label": label.lower(),
                        "confidence": float(confidences[idx]),
                        "box": [x1, y1, x2, y2],
                    }
                )
        return detections

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self.cap.release()

    def _capture_loop(self):
        # We target ~15 FPS. Inference usually takes some time, so loop delay can be minimal.
        failed_reads = 0
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                failed_reads += 1
                if failed_reads % 30 == 0:
                    print("[CAMERA] Frame read failed. Attempting camera reconnect...")
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = self._open_camera() or self.cap

                # Keep UI alive with a placeholder frame when camera read fails.
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "CAMERA READ FAILED", (120, 220), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.putText(placeholder, "Check USB webcam /dev/video0", (95, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 200), 1)
                ok, buffer = cv2.imencode('.jpg', placeholder)
                if ok:
                    with self._lock:
                        self._current_frame = buffer.tobytes()
                        self._current_confidence = 0.0
                        self._vision_detected = False
                time.sleep(0.1)
                continue
            failed_reads = 0

            highest_conf = 0.0
            best_box = None
            label_text = ""

            if self.vision_enabled:
                detections = []
                if self._backend == "opencv_onnx":
                    try:
                        detections = self._infer_opencv_onnx(frame)
                    except Exception as e:
                        print(f"[CAMERA] ONNX inference error: {e}")
                        detections = []

                for det in detections:
                    label = det["label"]
                    if label not in self.proxy_labels:
                        continue
                    conf = det["confidence"]
                    if conf > highest_conf:
                        highest_conf = conf
                        best_box = det["box"]
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
                self._vision_detected = highest_conf >= self.VISION_THRESHOLD

            # Print detection output to console
            if self.vision_enabled:
                status = "DRONE DETECTED" if self._vision_detected else "No Drone"
                print(f"[CAMERA] Confidence: {highest_conf*100:5.1f}% | {status}")
                
            # Yield briefly to not peg CPU 100%
            time.sleep(0.01)

    def get_latest(self):
        with self._lock:
            return {
                "frame": self._current_frame,
                "confidence": self._current_confidence,
                "vision_detected": self._vision_detected,
                "vision_enabled": self.vision_enabled
            }
