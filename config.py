# config.py

class CameraConfig:
    WIDTH = 1280
    HEIGHT = 720
    FPS = 15

class AudioConfig:
    SAMPLE_RATE = 44100
    CHUNK_SIZE = 1024

class SystemConfig:
    FUSION_THRESHOLD = 0.75
    FUSION_VISION_WEIGHT = 0.6
    FUSION_AUDIO_WEIGHT = 0.4
    SMOOTHING_FACTOR = 0.2  # For Exponential Moving Average
