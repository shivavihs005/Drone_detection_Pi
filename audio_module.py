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

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def _capture_loop(self):
        print("[AUDIO] Starting ALSA capture from plughw:1,0...")
        
        # Use arecord to pull robust ALSA stream from plughw:1,0
        arecord_cmd = [
            "arecord", "-D", "plughw:1,0", "-f", "S16_LE", "-r", str(self.RATE), "-c", "1", "-t", "raw"
        ]
        
        try:
            process = subprocess.Popen(arecord_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[AUDIO] Failed to start arecord: {e}")
            process = None

        while self._running:
            if process:
                raw_data = process.stdout.read(self.CHUNK_BYTES)
                if len(raw_data) == self.CHUNK_BYTES:
                    audio_data = np.frombuffer(raw_data, dtype=np.int16)
                else:
                    audio_data = np.zeros(int(self.RATE * self.CHUNK_DURATION), dtype=np.int16)
            else:
                # Mock fallback if ALSA fails or tested on windows
                time.sleep(self.CHUNK_DURATION)
                audio_data = np.random.normal(0, 100, int(self.RATE * self.CHUNK_DURATION))

            try:
                # 1. Bandpass filter 80 Hz to 4000 Hz
                filtered_data = self.butter_bandpass_filter(audio_data, 80, 4000, self.RATE, order=3)
                
                # 2. Compute FFT
                fft_result = np.fft.rfft(filtered_data)
                fft_freqs = np.fft.rfftfreq(len(filtered_data), 1.0/self.RATE)
                magnitudes = np.abs(fft_result)
                
                # 3. Find dominant freq
                peak_idx = np.argmax(magnitudes)
                peak_freq = fft_freqs[peak_idx]
                peak_mag = magnitudes[peak_idx]
                
                confidence = 0.0
                
                # 4. Check if peak is between 100-600 Hz
                if 100 <= peak_freq <= 600:
                    mean_mag = np.mean(magnitudes)
                    
                    # Normalize peak relative to noise floor
                    peak_ratio = peak_mag / (mean_mag + 1e-6)
                    
                    if peak_ratio > 5.0:  # strong narrowband tone
                        confidence = 0.4
                        
                        # Note: Simple explicit checks for harmonics
                        h2_target = peak_freq * 2
                        h3_target = peak_freq * 3
                        
                        h2_idx = np.argmin(np.abs(fft_freqs - h2_target))
                        h3_idx = np.argmin(np.abs(fft_freqs - h3_target))
                        
                        if magnitudes[h2_idx] > mean_mag * 2:
                            confidence += 0.3
                        if magnitudes[h3_idx] > mean_mag * 2:
                            confidence += 0.3
                            
                # Clamp to 1.0
                confidence = min(max(confidence, 0.0), 1.0)
                
            except Exception as e:
                print(f"[AUDIO] DSP Error: {e}")
                peak_freq = 0.0
                confidence = 0.0

            with self._lock:
                self._current_confidence = confidence
                self._dominant_freq = int(peak_freq)

        if process:
            process.terminate()

    def get_latest(self):
        with self._lock:
            return {
                "dominant_freq": self._dominant_freq,
                "confidence": self._current_confidence
            }
