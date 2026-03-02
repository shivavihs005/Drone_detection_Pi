import argparse
import json
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


LABEL_TO_ID = {"background": 0, "drone": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train drone-vs-background audio classifier"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate for loading audio (default: 16000)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Audio clip duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("weights/audio_rf_model.joblib"),
        help="Path to save trained model (default: weights/audio_rf_model.joblib)",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("runs/audio_train_report.json"),
        help="Path to save training report (default: runs/audio_train_report.json)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="RandomForest trees (default: 300)",
    )
    return parser.parse_args()


def list_audio_files(folder: Path) -> list[Path]:
    supported = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in supported]


def featurize_file(path: Path, sample_rate: int, duration: float) -> np.ndarray:
    target_len = int(sample_rate * duration)
    audio, _ = librosa.load(path, sr=sample_rate, mono=True, duration=duration)

    if audio.shape[0] < target_len:
        audio = np.pad(audio, (0, target_len - audio.shape[0]))
    else:
        audio = audio[:target_len]

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    rms = librosa.feature.rms(y=audio)

    features = np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.array(
                [
                    np.mean(spectral_centroid),
                    np.std(spectral_centroid),
                    np.mean(spectral_bandwidth),
                    np.std(spectral_bandwidth),
                    np.mean(rolloff),
                    np.std(rolloff),
                    np.mean(zcr),
                    np.std(zcr),
                    np.mean(rms),
                    np.std(rms),
                ]
            ),
        ]
    )
    return features.astype(np.float32)


def load_split(root: Path, split: str, sample_rate: int, duration: float) -> tuple[np.ndarray, np.ndarray]:
    x_items: list[np.ndarray] = []
    y_items: list[int] = []

    for label_name, label_id in LABEL_TO_ID.items():
        folder = root / "datasets" / "audio" / split / label_name
        if not folder.exists():
            continue
        for audio_file in list_audio_files(folder):
            try:
                x_items.append(featurize_file(audio_file, sample_rate, duration))
                y_items.append(label_id)
            except Exception as exc:
                print(f"Skipping {audio_file}: {exc}")

    if not x_items:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.vstack(x_items), np.array(y_items, dtype=np.int32)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    print("Loading audio features...")
    x_train, y_train = load_split(root, "train", args.sample_rate, args.duration)
    x_val, y_val = load_split(root, "val", args.sample_rate, args.duration)
    x_test, y_test = load_split(root, "test", args.sample_rate, args.duration)

    if x_train.size == 0:
        raise RuntimeError("No training audio found in datasets/audio/train")
    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Training data must contain both 'drone' and 'background' classes")

    print(f"Train samples: {len(y_train)}")
    print(f"Val samples: {len(y_val)}")
    print(f"Test samples: {len(y_test)}")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    def evaluate(split_name: str, x_data: np.ndarray, y_data: np.ndarray) -> dict:
        if x_data.size == 0:
            return {"samples": 0, "accuracy": None, "report": None, "confusion_matrix": None}

        y_pred = model.predict(x_data)
        acc = float(accuracy_score(y_data, y_pred))
        report = classification_report(
            y_data,
            y_pred,
            target_names=["background", "drone"],
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_data, y_pred).tolist()
        print(f"{split_name} accuracy: {acc:.4f}")
        return {
            "samples": int(len(y_data)),
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm,
        }

    val_metrics = evaluate("Validation", x_val, y_val)
    test_metrics = evaluate("Test", x_test, y_test)

    model_path = (root / args.model_out).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "sample_rate": args.sample_rate,
            "duration": args.duration,
            "labels": LABEL_TO_ID,
        },
        model_path,
    )

    report_path = (root / args.report_out).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "train_samples": int(len(y_train)),
        "val": val_metrics,
        "test": test_metrics,
        "model_path": str(model_path),
        "sample_rate": args.sample_rate,
        "duration": args.duration,
        "n_estimators": args.n_estimators,
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print(f"Model saved: {model_path}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
