import threading
import time
import numpy as np
from scipy.signal import butter, lfilter
import subprocess

class AudioModule:
    def __init__(self):
        self._thread = None
        self._running = False
        self._current_confidence = 0.0
        self._dominant_freq = 0
        self._lock = threading.Lock()
        self.audio_enabled = True
        
        self.RATE = 48000
        self.CHUNK_DURATION = 1.0
        
        # 16-bit audio = 2 bytes per sample
        self.CHUNK_BYTES = int(self.RATE * self.CHUNK_DURATION * 2)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def audio_drone_score(self, chunk: np.ndarray, sample_rate: int) -> tuple[float, int]:
        """Calculate drone likelihood score from audio chunk (Ported from drone_detection.py)."""
        if chunk.size == 0:
            return 0.0, 0

        signal = chunk.astype(np.float32)
        signal = signal - np.mean(signal)
        rms = float(np.sqrt(np.mean(signal**2)))

        if rms < 0.001:
            return 0.0, 0

        window = np.hanning(signal.shape[0]).astype(np.float32)
        spectrum = np.fft.rfft(signal * window)
        freqs = np.fft.rfftfreq(signal.shape[0], d=1.0 / sample_rate)
        power = np.abs(spectrum) ** 2

        if power.size == 0:
            return 0.0, 0
            
        peak_idx = np.argmax(power)
        peak_freq = int(freqs[peak_idx])

        total_power = float(np.sum(power)) + 1e-8
        drone_band = (freqs >= 100) & (freqs <= 1200)
        drone_band_energy = float(np.sum(power[drone_band])) / total_power

        score = 0.6 * drone_band_energy + 0.4 * min(rms / 0.05, 1.0)
        return float(np.clip(score, 0.0, 1.0)), peak_freq

    def _capture_loop(self):
        print("[AUDIO] Starting ALSA capture from plughw:1,0...")
        
        # Use arecord to pull robust ALSA stream from plughw:1,0 using S24_3LE format
        arecord_cmd = [
            "arecord", "-D", "plughw:1,0", "-f", "S24_3LE", "-r", str(self.RATE), "-c", "1", "-t", "raw"
        ]
        
        # 24-bit audio = 3 bytes per sample
        self.CHUNK_BYTES = int(self.RATE * self.CHUNK_DURATION * 3)
        
        try:
            process = subprocess.Popen(arecord_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[AUDIO] Failed to start arecord: {e}")
            process = None

        while self._running:
            if process:
                raw_data = process.stdout.read(self.CHUNK_BYTES)
                if len(raw_data) == self.CHUNK_BYTES:
                    # Convert 3-byte 24-bit PCM (S24_3LE) to 32-bit integers using numpy
                    padded_data = np.zeros(len(raw_data) // 3 * 4, dtype=np.uint8)
                    padded_data[0::4] = np.frombuffer(raw_data[0::3], dtype=np.uint8)
                    padded_data[1::4] = np.frombuffer(raw_data[1::3], dtype=np.uint8)
                    padded_data[2::4] = np.frombuffer(raw_data[2::3], dtype=np.uint8)
                    padded_data[3::4] = (padded_data[2::4] > 127) * 255
                    audio_data = padded_data.view(np.int32)
                else:
                    audio_data = np.zeros(int(self.RATE * self.CHUNK_DURATION), dtype=np.int32)
            else:
                # Mock fallback if ALSA fails or tested on windows
                time.sleep(self.CHUNK_DURATION)
                audio_data = np.random.normal(0, 100, int(self.RATE * self.CHUNK_DURATION))

            # Skip DSP Processing if the module is turned OFF
            if not self.audio_enabled:
                with self._lock:
                    self._current_confidence = 0.0
                    self._dominant_freq = 0
                continue

            try:
                confidence, peak_freq = self.audio_drone_score(audio_data, self.RATE)
            except Exception as e:
                print(f"[AUDIO] DSP Error: {e}")
                peak_freq = 0
                confidence = 0.0

            with self._lock:
                # Smooth the audio scores to prevent jittering UI
                self._current_confidence = 0.7 * self._current_confidence + 0.3 * confidence
                self._dominant_freq = int(peak_freq)

        if process:
            process.terminate()

    def get_latest(self):
        with self._lock:
            return {
                "dominant_freq": self._dominant_freq,
                "confidence": self._current_confidence,
                "audio_enabled": self.audio_enabled
            }
