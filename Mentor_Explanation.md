# Drone Detection System: Technical Overview for Mentor

This document outlines the core technical processes of the multimodal drone detection system running on the Raspberry Pi 5.

## 1. How the Data is Read (Raspberry Pi)

The system is designed to read two streams of data simultaneously using background threading so the main server remains responsive.

*   **Image Data (Vision):** The system uses `picamera2`, the official Raspberry Pi library, to continually capture 720p video frames directly from the camera hardware. If a Pi camera isn't detected (e.g., during testing), it automatically falls back to reading from a standard USB webcam using `OpenCV`.
*   **Audio Data (Sound):** The system bypasses standard high-level sound libraries and uses the native Linux `arecord` command line tool to reliably pull raw audio buffers from the ALSA sound driver (`plughw:1,0`). It captures high-quality 24-bit audio at 48kHz in small chunks.

## 2. How the Data is Compared and Fused

Once the data is read, the system analyzes and compares the vision and audio streams to make a final decision.

*   **Vision Analysis:** The system passes the captured image to a YOLOv8 neural network. The neural network scans the image for visual patterns matching a drone and outputs a **Vision Confidence Score** (0.0 to 1.0) and bounding box coordinates.
*   **Audio Analysis:** The raw audio chunk is mathematically transformed from a time waveform into a frequency graph using a Fast Fourier Transform (`np.fft`). The system calculates how much energy is concentrated within the typical "drone buzz" frequency band (100Hz to 1200Hz) and outputs an **Audio Confidence Score**.
*   **The Fusion Algorithm:** The `FusionModule` combines these scores. Because vision is generally more reliable for pinpointing exact locations, it is given a higher weight (70%) compared to audio (30%). If the mathematically combined score (`0.7 * vision + 0.3 * audio`) exceeds a strict threshold, the system flags a positive detection.

## 3. How the Model is Trained

The vision model is not built from scratch; it uses Transfer Learning.

*   **Base Model:** We begin with `yolov8n.pt`, a pre-trained "nano" YOLO model built by Ultralytics. It's incredibly lightweight and designed specifically for edge devices like the Raspberry Pi.
*   **Custom Dataset:** We feed the model a dataset containing images of drones to teach it specifically what drones look like in various environments.
*   **Fast Prototype Training:** To maximize frame rate on the Pi, we train the model using a reduced image resolution (`imgsz=320` instead of the standard 640). During training, the model repeatedly guesses where the drone is, checks the actual answer, and adjusts its internal neural weights.
*   **Export:** The training process outputs a final weights file (`best.pt`) which is what the Pi loads into memory at runtime.

## 4. What the Raspberry Pi Configuration Is

The Pi environment is specifically configured to balance isolated Python dependencies with native hardware access.

*   **Hardware:** A Raspberry Pi 5 connected to a compatible camera module and a USB microphone.
*   **OS & Dependencies:** The system utilizes native UNIX tools like `alsa-utils` and `libcamera-apps` to ensure direct hardware communication.
*   **Python Virtual Environment:** The entire application runs inside an isolated Python virtual environment (`drone_env`). Crucially, this environment is created using the `--system-site-packages` flag. This allows our isolated Python setup to still interact with the deeply integrated Pi hardware drivers (like `picamera2`).
*   **Web Server:** The resulting application runs a lightweight `Flask` web server on port 5000, streaming the live video and metrics dashboard to any device on the local network.
