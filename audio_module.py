import threading
import time
import numpy as np
import os

# Try sounddevice (cross-platform: Windows, Mac, Linux)
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

# Try to load ML libraries for trained model inference
try:
    import librosa
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Fall back to subprocess arecord only on Linux / Raspberry Pi
import subprocess
import platform


class AudioModule:
    def __init__(self):
        self._thread = None
        self._running = False
        self._current_confidence = 0.0
        self._dominant_freq = 0
        self._audio_detected = False
        self._lock = threading.Lock()
        self.audio_enabled = True
        self.AUDIO_THRESHOLD = 0.5  # confidence threshold for audio-only drone detection

        self.RATE = 44100
        self.CHUNK_DURATION = 1.0  # seconds per analysis window
        self.CHUNK_SIZE = int(self.RATE * self.CHUNK_DURATION)

        # --- Load trained audio model if available ---
        self._ml_model = None
        self._ml_sr = 22050
        self._ml_n_mfcc = 20
        model_path = 'models/audio_drone_model.joblib'
        if ML_AVAILABLE and os.path.exists(model_path):
            try:
                data = joblib.load(model_path)
                self._ml_model = data['model']
                self._ml_sr = data.get('sample_rate', 22050)
                self._ml_n_mfcc = data.get('n_mfcc', 20)
                print(f"[AUDIO] Trained audio model loaded from {model_path}")
            except Exception as e:
                print(f"[AUDIO] Failed to load trained model: {e}. Using DSP fallback.")
        else:
            if not ML_AVAILABLE:
                print("[AUDIO] librosa/joblib not installed. Using DSP-based detection.")
            else:
                print("[AUDIO] No trained model found. Run train_audio_model.py first. Using DSP fallback.")

        # Decide capture backend
        if SOUNDDEVICE_AVAILABLE:
            self._backend = "sounddevice"
            print("[AUDIO] Using sounddevice backend (cross-platform).")
        elif platform.system() == "Linux":
            self._backend = "arecord"
            print("[AUDIO] sounddevice not found. Falling back to arecord (Linux/Pi only).")
        else:
            self._backend = "mock"
            print("[AUDIO] No audio capture backend available. Using mock audio (random noise).")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def get_latest(self):
        with self._lock:
            return {
                "dominant_freq": self._dominant_freq,
                "confidence": self._current_confidence,
                "audio_detected": self._audio_detected,
                "audio_enabled": self.audio_enabled,
            }

    # ------------------------------------------------------------------
    # DSP: drone likelihood score
    # ------------------------------------------------------------------

    def audio_drone_score(self, chunk: np.ndarray, sample_rate: int) -> tuple:
        """Calculate drone likelihood score from audio chunk.
        Uses trained ML model if available, otherwise falls back to DSP."""
        if chunk.size == 0:
            return 0.0, 0

        signal = chunk.astype(np.float32)
        signal = signal - np.mean(signal)
        rms = float(np.sqrt(np.mean(signal ** 2)))

        if rms < 0.001:
            return 0.0, 0

        # Compute peak frequency (used by both paths)
        window = np.hanning(signal.shape[0]).astype(np.float32)
        spectrum = np.fft.rfft(signal * window)
        freqs = np.fft.rfftfreq(signal.shape[0], d=1.0 / sample_rate)
        power = np.abs(spectrum) ** 2

        if power.size == 0:
            return 0.0, 0

        peak_idx = np.argmax(power)
        peak_freq = int(freqs[peak_idx])

        # --- ML model path ---
        if self._ml_model is not None:
            try:
                score = self._predict_ml(signal, sample_rate)
                return float(np.clip(score, 0.0, 1.0)), peak_freq
            except Exception:
                pass  # fall through to DSP

        # --- DSP fallback ---
        total_power = float(np.sum(power)) + 1e-8
        drone_band = (freqs >= 100) & (freqs <= 1200)
        drone_band_energy = float(np.sum(power[drone_band])) / total_power

        score = 0.6 * drone_band_energy + 0.4 * min(rms / 0.05, 1.0)
        return float(np.clip(score, 0.0, 1.0)), peak_freq

    def _predict_ml(self, signal: np.ndarray, sample_rate: int) -> float:
        """Run the trained ML model on an audio chunk."""
        # Resample to the model's expected sample rate if needed
        if sample_rate != self._ml_sr:
            signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=self._ml_sr)

        mfccs = librosa.feature.mfcc(y=signal, sr=self._ml_sr, n_mfcc=self._ml_n_mfcc)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=signal, sr=self._ml_sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=self._ml_sr))
        spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=self._ml_sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(signal))

        features = np.concatenate([mfcc_mean, mfcc_std, [spec_cent, spec_bw, spec_rolloff, zcr]])
        proba = self._ml_model.predict_proba(features.reshape(1, -1))[0]
        # Return probability of drone class (class 1)
        return float(proba[1])

    # ------------------------------------------------------------------
    # Capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self):
        if self._backend == "sounddevice":
            self._loop_sounddevice()
        elif self._backend == "arecord":
            self._loop_arecord()
        else:
            self._loop_mock()

    # ---- sounddevice (Windows / Mac / Linux) -------------------------

    def _loop_sounddevice(self):
        print("[AUDIO] sounddevice capture started.")
        try:
            # List available input devices for debugging
            device_info = sd.query_devices(kind='input')
            print(f"[AUDIO] Default input device: {device_info['name']}")
        except Exception as e:
            print(f"[AUDIO] Could not query devices: {e}")

        while self._running:
            if not self.audio_enabled:
                with self._lock:
                    self._current_confidence = 0.0
                    self._dominant_freq = 0
                time.sleep(0.1)
                continue

            try:
                # Record CHUNK_DURATION seconds from the default microphone
                recording = sd.rec(
                    self.CHUNK_SIZE,
                    samplerate=self.RATE,
                    channels=1,
                    dtype="float32",
                    blocking=True,
                )
                audio_data = recording[:, 0]  # mono
            except Exception as e:
                print(f"[AUDIO] sounddevice capture error: {e}")
                time.sleep(0.5)
                continue

            self._process(audio_data)

    # ---- arecord (Linux / Raspberry Pi) ------------------------------

    def _loop_arecord(self):
        print("[AUDIO] Starting ALSA capture from plughw:1,0...")
        rate = 48000
        chunk_bytes = int(rate * self.CHUNK_DURATION * 3)  # S24_3LE = 3 bytes/sample

        arecord_cmd = [
            "arecord", "-D", "plughw:1,0",
            "-f", "S24_3LE", "-r", str(rate), "-c", "1", "-t", "raw",
        ]

        try:
            process = subprocess.Popen(
                arecord_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"[AUDIO] Failed to start arecord: {e}")
            process = None

        while self._running:
            if not self.audio_enabled:
                with self._lock:
                    self._current_confidence = 0.0
                    self._dominant_freq = 0
                time.sleep(0.1)
                continue

            if process:
                raw_data = process.stdout.read(chunk_bytes)
                if len(raw_data) == chunk_bytes:
                    # Convert S24_3LE → int32 → float32
                    padded = np.zeros(len(raw_data) // 3 * 4, dtype=np.uint8)
                    padded[0::4] = np.frombuffer(raw_data[0::3], dtype=np.uint8)
                    padded[1::4] = np.frombuffer(raw_data[1::3], dtype=np.uint8)
                    padded[2::4] = np.frombuffer(raw_data[2::3], dtype=np.uint8)
                    padded[3::4] = (padded[2::4] > 127) * 255
                    audio_data = padded.view(np.int32).astype(np.float32)
                else:
                    audio_data = np.zeros(int(rate * self.CHUNK_DURATION), dtype=np.float32)
            else:
                time.sleep(self.CHUNK_DURATION)
                audio_data = np.zeros(int(rate * self.CHUNK_DURATION), dtype=np.float32)

            self._process(audio_data, sample_rate=rate)

        if process:
            process.terminate()

    # ---- mock / silent fallback -------------------------------------

    def _loop_mock(self):
        print("[AUDIO] Mock mode: generating synthetic audio (no real mic).")
        while self._running:
            time.sleep(self.CHUNK_DURATION)
            if not self.audio_enabled:
                with self._lock:
                    self._current_confidence = 0.0
                    self._dominant_freq = 0
                continue
            # Mild white-noise so UI shows some activity in mock mode
            audio_data = np.random.normal(0, 100, self.CHUNK_SIZE).astype(np.float32)
            self._process(audio_data)

    # ------------------------------------------------------------------
    # Shared processing helper
    # ------------------------------------------------------------------

    def _process(self, audio_data: np.ndarray, sample_rate: int = None):
        if sample_rate is None:
            sample_rate = self.RATE
        try:
            confidence, peak_freq = self.audio_drone_score(audio_data, sample_rate)
        except Exception as e:
            print(f"[AUDIO] DSP Error: {e}")
            confidence, peak_freq = 0.0, 0

        with self._lock:
            # Exponential smoothing to reduce jitter in the UI
            self._current_confidence = 0.7 * self._current_confidence + 0.3 * confidence
            self._dominant_freq = int(peak_freq)
            self._audio_detected = self._current_confidence >= self.AUDIO_THRESHOLD

        # Print detection output to console so user sees results on the Pi
        status = "DRONE DETECTED" if self._audio_detected else "No Drone"
        conf_pct = self._current_confidence * 100
        model_tag = "ML" if self._ml_model is not None else "DSP"
        print(f"[AUDIO] [{model_tag}] Confidence: {conf_pct:5.1f}% | Freq: {peak_freq:5d} Hz | {status}")
