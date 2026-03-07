"""
Train a drone audio detection model using the first 10 drone audio files
and 10 non-drone audio files from the Binary_Drone_Audio dataset.

Extracts MFCC features with librosa and trains a Random Forest classifier.
Saves the model to models/audio_drone_model.joblib

Usage:
    python train_audio_model.py
"""

import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib

# --- Configuration ---
DRONE_DIR = 'DroneAudioDataset-master/Binary_Drone_Audio/yes_drone'
NON_DRONE_DIR = 'DroneAudioDataset-master/Binary_Drone_Audio/unknown'
MODEL_OUTPUT = 'models/audio_drone_model.joblib'
NUM_DRONE_FILES = 10
NUM_NON_DRONE_FILES = 10
SAMPLE_RATE = 22050
N_MFCC = 20


def extract_features(file_path, sr=SAMPLE_RATE):
    """Extract MFCC features from a single audio file."""
    audio, sr = librosa.load(file_path, sr=sr)

    # Skip very short/silent files
    if len(audio) < sr * 0.1:
        return None

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

    # Use mean and std of each MFCC coefficient as features
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    # Spectral centroid and bandwidth
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

    features = np.concatenate([mfcc_mean, mfcc_std, [spec_cent, spec_bw, spec_rolloff, zcr]])
    return features


def load_audio_files(directory, max_files):
    """Load and sort wav files from a directory, return first max_files."""
    all_files = sorted([f for f in os.listdir(directory) if f.endswith('.wav')])
    selected = all_files[:max_files]
    print(f"  Selected {len(selected)} files from {directory}")
    for f in selected:
        print(f"    - {f}")
    return [os.path.join(directory, f) for f in selected]


def main():
    print("=" * 50)
    print(" Drone Audio Model Training")
    print("=" * 50)

    os.makedirs('models', exist_ok=True)

    # Load file paths
    print(f"\n[1/4] Loading first {NUM_DRONE_FILES} drone audio files...")
    drone_files = load_audio_files(DRONE_DIR, NUM_DRONE_FILES)

    print(f"\n[2/4] Loading first {NUM_NON_DRONE_FILES} non-drone audio files...")
    non_drone_files = load_audio_files(NON_DRONE_DIR, NUM_NON_DRONE_FILES)

    # Extract features
    print("\n[3/4] Extracting MFCC features...")
    X = []
    y = []

    for f in drone_files:
        features = extract_features(f)
        if features is not None:
            X.append(features)
            y.append(1)  # drone
            print(f"  [DRONE]     {os.path.basename(f)} -> {len(features)} features")

    for f in non_drone_files:
        features = extract_features(f)
        if features is not None:
            X.append(features)
            y.append(0)  # not drone
            print(f"  [NON-DRONE] {os.path.basename(f)} -> {len(features)} features")

    X = np.array(X)
    y = np.array(y)

    print(f"\n  Total samples: {len(X)} ({sum(y)} drone, {len(y) - sum(y)} non-drone)")
    print(f"  Feature vector size: {X.shape[1]}")

    # Train model
    print("\n[4/4] Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    # Cross-validation if we have enough samples
    if len(X) >= 4:
        n_splits = min(5, len(X))
        scores = cross_val_score(clf, X, y, cv=n_splits, scoring='accuracy')
        print(f"  Cross-validation accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

    # Train on all data
    clf.fit(X, y)

    # Save model
    joblib.dump({
        'model': clf,
        'n_mfcc': N_MFCC,
        'sample_rate': SAMPLE_RATE,
    }, MODEL_OUTPUT)

    print(f"\n  Model saved to: {MODEL_OUTPUT}")
    print("=" * 50)
    print(" Training complete!")
    print("=" * 50)


if __name__ == '__main__':
    main()
