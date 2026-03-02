import argparse
import queue
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import sounddevice as sd
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time drone-like object detection from webcam"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help=(
            "Path to YOLO model. Use a custom drone model (best) or the default "
            "COCO model as a proxy."
        ),
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
        help=(
            "Comma-separated class labels treated as drone in custom model "
            "(default: drone,uav,quadcopter)"
        ),
    )
    parser.add_argument(
        "--audio-device",
        type=int,
        default=None,
        help="Microphone device index for audio detection (default: system default)",
    )
    parser.add_argument(
        "--audio-threshold",
        type=float,
        default=0.18,
        help="Audio drone-likelihood threshold between 0 and 1 (default: 0.18)",
    )
    return parser.parse_args()


def audio_drone_score(chunk: np.ndarray, sample_rate: int) -> float:
    if chunk.size == 0:
        return 0.0

    signal = chunk.astype(np.float32)
    signal = signal - np.mean(signal)
    rms = float(np.sqrt(np.mean(signal**2)))

    if rms < 0.001:
        return 0.0

    window = np.hanning(signal.shape[0]).astype(np.float32)
    spectrum = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(signal.shape[0], d=1.0 / sample_rate)
    power = np.abs(spectrum) ** 2

    if power.size == 0:
        return 0.0

    total_power = float(np.sum(power)) + 1e-8
    drone_band = (freqs >= 100) & (freqs <= 1200)
    drone_band_energy = float(np.sum(power[drone_band])) / total_power

    score = 0.6 * drone_band_energy + 0.4 * min(rms / 0.05, 1.0)
    return float(np.clip(score, 0.0, 1.0))


def main() -> None:
    args = parse_args()

    if not Path(args.model).exists() and args.model != "yolov8n.pt":
        raise FileNotFoundError(f"Model file not found: {args.model}")

    model = YOLO(args.model)
    drone_labels = {name.strip().lower() for name in args.drone_labels.split(",") if name}

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. Try --camera 1 or 2."
        )

    # COCO labels used only as rough proxy when no custom drone model is provided.
    proxy_labels = {"airplane", "helicopter", "bird"}
    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
    sample_rate = 16000
    audio_score = 0.0

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        del frames, time_info
        if status:
            return
        mono = indata[:, 0].copy()
        if audio_queue.full():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                pass
        audio_queue.put_nowait(mono)

    audio_stream: Optional[sd.InputStream] = None
    try:
        audio_stream = sd.InputStream(
            device=args.audio_device,
            channels=1,
            samplerate=sample_rate,
            blocksize=2048,
            callback=audio_callback,
            dtype="float32",
        )
        audio_stream.start()
        print("Audio detection enabled.")
    except Exception as exc:
        print(f"Audio detection disabled: {exc}")

    print("Starting detection... Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(source=frame, conf=args.conf, verbose=False)
        result = results[0]

        names = result.names
        boxes = result.boxes
        vision_score = 0.0
        detected_count = 0

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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(
                frame,
                f"{display_label} {score:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 0),
                2,
            )

        while not audio_queue.empty():
            try:
                chunk = audio_queue.get_nowait()
            except queue.Empty:
                break
            new_audio_score = audio_drone_score(chunk, sample_rate)
            audio_score = 0.7 * audio_score + 0.3 * new_audio_score

        audio_hit = audio_score >= args.audio_threshold
        fused_score = float(np.clip(0.65 * vision_score + 0.35 * audio_score, 0.0, 1.0))
        alert = detected_count > 0 and (vision_score >= args.conf or audio_hit)

        status_color = (0, 0, 255) if alert else (0, 200, 255)
        status_text = "ALERT: DRONE LIKELY" if alert else "Monitoring"
        cv2.putText(
            frame,
            status_text,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )
        cv2.putText(
            frame,
            f"Vision:{vision_score:.2f} Audio:{audio_score:.2f} Fused:{fused_score:.2f}",
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Detections:{detected_count}  AudioHit:{audio_hit}",
            (12, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Drone Detection (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if audio_stream is not None:
        audio_stream.stop()
        audio_stream.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
