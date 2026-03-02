# Drone_detection_Pi

A full Raspberry Pi 5 multimodal drone detection system using Camera (YOLOv8) and Audio (DSP/FFT) fusion.

## Full Raspberry Pi Setup Walkthrough

This repository contains a unified setup script (`setup_pi.sh`) that will automatically:
1. Update system dependencies for camera and audio.
2. Pull the latest code from GitHub.
3. Create a Python Virtual Environment (`drone_env`) with system packages attached.
4. Activate the environment.
5. Install all required Python packages (Flask, OpenCV, Ultralytics YOLO, SciPy, NumPy).
6. Automatically run the main Flask server.

### Step 1: Clone the Repository to your Raspberry Pi
Open your Raspberry Pi terminal and run this command to download the code:
```bash
git clone https://github.com/shivavihs005/Drone_detection_Pi.git
cd Drone_detection_Pi
```

### Step 2: Make the Setup Script Executable
Give your Raspberry Pi permission to run the setup script:
```bash
chmod +x setup_pi.sh
```

### Step 3: Run the Setup Script
Run the script to automatically install everything, update the repo, and start the system:
```bash
./setup_pi.sh
```

*Note: The script might ask for your Pi password (`sudo`) to install system packages like `alsa-utils` and `libcamera-apps`.*

---

## What the Script Does (Behind the Scenes)
If you ever want to run these steps manually, here is exactly what the `setup_pi.sh` script does:

1. **System Packages**: `sudo apt-get install -y python3-venv python3-pip alsa-utils libcamera-apps` (Ensures you have the audio recording tools and camera libraries).
2. **GitHub Pull**: `git pull origin main` (Downloads any new code you pushed to GitHub from your PC).
3. **Virtual Environment**: 
   ```bash
   python3 -m venv --system-site-packages drone_env
   source drone_env/bin/activate
   ```
   *(Using `--system-site-packages` is critical on the Pi so that the virtual environment can still talk to the Pi's native camera drivers!)*
4. **Install Python Libraries**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Run the Server**:
   ```bash
   python3 main_server.py
   ```

## Viewing the Web Interface
Once the setup script finishes and prints `[SYSTEM] System initialized successfully.`:
1. Open a web browser on any device connected to the same Wi-Fi.
2. Go to `http://<YOUR_PI_IP_ADDRESS>:5000` (e.g., `http://192.168.1.100:5000`).
3. You will see the local dashboard streaming the camera and showing detection confidence!



cd /home/nitish/Drone_detection_Pi
source drone_env/bin/activate
python3 main_server.py

