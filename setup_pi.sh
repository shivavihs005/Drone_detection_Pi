#!/bin/bash

echo "=========================================="
echo " Starting Drone Detection Pi Setup Script"
echo "=========================================="

# 1. Update system dependencies
echo ">>> Checking/Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-venv \
    python3-pip \
    python3-dev \
    alsa-utils \
    libportaudio2 \
    libsndfile1 \
    v4l-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg

# 2. Verify USB webcam is detected
echo ">>> Checking for USB webcam..."
if v4l2-ctl --list-devices 2>/dev/null | grep -q "video"; then
    echo ">>> USB webcam detected:"
    v4l2-ctl --list-devices
else
    echo ">>> WARNING: No USB webcam detected. Please connect one before running the server."
fi

# 3. Update the repository from GitHub
echo ">>> Pulling latest code from GitHub..."
git pull origin main

# 4. Create virtual environment
ENV_DIR="drone_env"
if [ ! -d "$ENV_DIR" ]; then
    echo ">>> Creating virtual environment in $ENV_DIR..."
    python3 -m venv --system-site-packages $ENV_DIR
else
    echo ">>> Virtual environment already exists."
fi

# 5. Activate virtual environment
echo ">>> Activating virtual environment..."
source $ENV_DIR/bin/activate

# 6. Install required Python packages
echo ">>> Installing Python dependencies..."
python3 -m pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo ">>> requirements.txt not found! Installing default packages..."
    pip install Flask numpy scipy ultralytics opencv-python sounddevice requests librosa joblib scikit-learn
fi

# 7. Create required directories
echo ">>> Creating required directories..."
mkdir -p models
mkdir -p templates

# 8. Run the main server
echo ""
echo "=========================================="
echo " Setup complete!"
echo " Using USB webcam for detection."
echo "=========================================="
echo ">>> Starting the Main Server..."
python3 main_server.py
