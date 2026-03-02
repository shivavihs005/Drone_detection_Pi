import argparse
import random
import shutil
from pathlib import Path


SUPPORTED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare train/val/test audio dataset splits for drone detection"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root path (default: current directory)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train split ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clear existing train/val/test audio files before splitting",
    )
    return parser.parse_args()


def list_audio_files(folder: Path) -> list[Path]:
    files: list[Path] = []
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXT:
            files.append(path)
    return files


def clear_folder(folder: Path) -> None:
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        return
    for item in folder.iterdir():
        if item.is_file():
            item.unlink()


def copy_split(files: list[Path], train_dir: Path, val_dir: Path, test_dir: Path, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    total = len(files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    train_files = files[:train_count]
    val_files = files[train_count : train_count + val_count]
    test_files = files[train_count + val_count :]

    for source in train_files:
        shutil.copy2(source, train_dir / source.name)
    for source in val_files:
        shutil.copy2(source, val_dir / source.name)
    for source in test_files:
        shutil.copy2(source, test_dir / source.name)

    return len(train_files), len(val_files), len(test_files)


def main() -> None:
    args = parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Invalid split ratios. Ensure train_ratio > 0, val_ratio >= 0 and train_ratio + val_ratio < 1.")

    root = args.project_root.resolve()
    raw_drone = root / "datasets" / "audio" / "raw" / "drone"
    raw_background = root / "datasets" / "audio" / "raw" / "background"

    if not raw_drone.exists() or not raw_background.exists():
        raise FileNotFoundError(
            "Missing raw audio folders. Expected datasets/audio/raw/drone and datasets/audio/raw/background"
        )

    split_dirs = {
        "train": {
            "drone": root / "datasets" / "audio" / "train" / "drone",
            "background": root / "datasets" / "audio" / "train" / "background",
        },
        "val": {
            "drone": root / "datasets" / "audio" / "val" / "drone",
            "background": root / "datasets" / "audio" / "val" / "background",
        },
        "test": {
            "drone": root / "datasets" / "audio" / "test" / "drone",
            "background": root / "datasets" / "audio" / "test" / "background",
        },
    }

    for split in split_dirs.values():
        for folder in split.values():
            folder.mkdir(parents=True, exist_ok=True)
            if args.clean:
                clear_folder(folder)

    drone_files = list_audio_files(raw_drone)
    background_files = list_audio_files(raw_background)

    if not drone_files:
        raise RuntimeError("No drone audio files found in datasets/audio/raw/drone")
    if not background_files:
        raise RuntimeError("No background audio files found in datasets/audio/raw/background")

    random.seed(args.seed)
    random.shuffle(drone_files)
    random.shuffle(background_files)

    drone_counts = copy_split(
        drone_files,
        split_dirs["train"]["drone"],
        split_dirs["val"]["drone"],
        split_dirs["test"]["drone"],
        args.train_ratio,
        args.val_ratio,
    )
    background_counts = copy_split(
        background_files,
        split_dirs["train"]["background"],
        split_dirs["val"]["background"],
        split_dirs["test"]["background"],
        args.train_ratio,
        args.val_ratio,
    )

    print("Audio dataset preparation complete")
    print(f"Drone clips     -> train:{drone_counts[0]} val:{drone_counts[1]} test:{drone_counts[2]}")
    print(
        f"Background clips-> train:{background_counts[0]} val:{background_counts[1]} test:{background_counts[2]}"
    )


if __name__ == "__main__":
    main()
