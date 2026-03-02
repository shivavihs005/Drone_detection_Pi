import argparse
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test drone audio detection")
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to audio file to test (optional, uses microphone if not provided)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds if using microphone (default: 5.0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.18,
        help="Drone detection threshold (default: 0.18)",
    )
    return parser.parse_args()


def audio_drone_score(chunk: np.ndarray, sample_rate: int) -> float:
    """Calculate drone likelihood score from audio chunk."""
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
    sample_rate = 16000

    if args.audio:
        # Test audio file
        audio_path = Path(args.audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {args.audio}")

        print(f"Loading audio file: {args.audio}")
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        
        # Split into 2-second chunks and analyze
        chunk_size = sample_rate * 2
        scores = []
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) < sample_rate * 0.5:  # Skip chunks less than 0.5 sec
                continue
            score = audio_drone_score(chunk, sample_rate)
            scores.append(score)
            print(f"Chunk {i//chunk_size + 1}: Score = {score:.4f}")
        
        avg_score = np.mean(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        
        print(f"\n{'='*50}")
        print(f"Average drone score: {avg_score:.4f}")
        print(f"Maximum drone score: {max_score:.4f}")
        print(f"Threshold: {args.threshold:.4f}")
        
        if avg_score >= args.threshold:
            print("✓ DRONE DETECTED (avg score above threshold)")
        else:
            print("✗ No drone detected (avg score below threshold)")
            
        if max_score >= args.threshold:
            print("✓ DRONE DETECTED in at least one chunk (max score above threshold)")
        else:
            print("✗ No drone detected in any chunk")
        print(f"{'='*50}")

    else:
        # Record from microphone
        print(f"Recording from microphone for {args.duration} seconds...")
        print("Make drone sounds now!")
        
        audio = sd.rec(
            int(args.duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        
        audio = audio[:, 0]  # Convert to mono
        
        # Analyze in 2-second windows
        chunk_size = sample_rate * 2
        scores = []
        
        print("\nAnalyzing recording...")
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) < sample_rate * 0.5:
                continue
            score = audio_drone_score(chunk, sample_rate)
            scores.append(score)
            timestamp = i / sample_rate
            print(f"Time {timestamp:.1f}s: Score = {score:.4f}")
        
        avg_score = np.mean(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        
        print(f"\n{'='*50}")
        print(f"Average drone score: {avg_score:.4f}")
        print(f"Maximum drone score: {max_score:.4f}")
        print(f"Threshold: {args.threshold:.4f}")
        
        if avg_score >= args.threshold:
            print("✓ DRONE DETECTED (avg score above threshold)")
        else:
            print("✗ No drone detected (avg score below threshold)")
            
        if max_score >= args.threshold:
            print("✓ DRONE DETECTED in at least one window (max score above threshold)")
        else:
            print("✗ No drone detected in any window")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
