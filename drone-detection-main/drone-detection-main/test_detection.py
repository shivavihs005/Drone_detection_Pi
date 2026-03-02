import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test drone detection on an image")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to test image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO model (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold (default: 0.35)",
    )
    parser.add_argument(
        "--drone-labels",
        type=str,
        default="drone,uav,quadcopter",
        help="Comma-separated drone labels (default: drone,uav,quadcopter)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="detection_result.jpg",
        help="Output image path (default: detection_result.jpg)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    if not Path(args.model).exists() and args.model != "yolov8n.pt":
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = YOLO(args.model)
    drone_labels = {name.strip().lower() for name in args.drone_labels.split(",") if name}
    proxy_labels = {"airplane", "helicopter", "bird"}

    frame = cv2.imread(args.image)
    if frame is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    results = model.predict(source=frame, conf=args.conf, verbose=False)
    result = results[0]

    names = result.names
    boxes = result.boxes

    detected_count = 0
    vision_score = 0.0

    for box in boxes:
        cls_idx = int(box.cls[0].item())
        label = names.get(cls_idx, str(cls_idx))
        score = float(box.conf[0].item())

        label_lower = label.lower()

        if args.model == "yolov8n.pt":
            if label_lower not in proxy_labels:
                continue
            display_label = f"possible_drone({label})"
        else:
            if label_lower not in drone_labels:
                continue
            display_label = "drone"

        vision_score = max(vision_score, score)
        detected_count += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 3)
        cv2.putText(
            frame,
            f"{display_label} {score:.2f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 0),
            2,
        )

    alert = detected_count > 0
    status_color = (0, 0, 255) if alert else (0, 200, 255)
    status_text = "ALERT: DRONE DETECTED" if alert else "NO DRONE"

    cv2.putText(
        frame,
        status_text,
        (15, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        status_color,
        3,
    )
    cv2.putText(
        frame,
        f"Vision Score: {vision_score:.2f} | Detections: {detected_count}",
        (15, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imwrite(args.output, frame)
    print(f"Result saved to: {args.output}")
    print(f"Detections: {detected_count}")
    print(f"Vision score: {vision_score:.2f}")
    print(f"Alert: {alert}")

    # Show the result
    cv2.imshow("Detection Result - Press any key to close", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
