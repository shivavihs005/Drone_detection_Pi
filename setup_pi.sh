#!/bin/bash

echo "=========================================="
echo " Starting Drone Detection Pi Setup Script"
echo "=========================================="

# 1. Update system dependencies 
echo ">>> Checking/Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip alsa-utils libcamera-apps

# 2. Update the repository from GitHub
echo ">>> Pulling latest code from GitHub..."
git pull origin main

# 3. Create virtual environment 
# Using --system-site-packages is crucial on Raspberry Pi to access picamera2
ENV_DIR="drone_env"
if [ ! -d "$ENV_DIR" ]; then
    echo ">>> Creating virtual environment in $ENV_DIR..."
    python3 -m venv --system-site-packages $ENV_DIR
else
    echo ">>> Virtual environment already exists."
fi

# 4. Activate virtual environment
echo ">>> Activating virtual environment..."
source $ENV_DIR/bin/activate

# 5. Install required Python packages
echo ">>> Installing Python dependencies..."
python3 -m pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo ">>> requirements.txt not found! Installing default packages..."
    pip install Flask numpy scipy ultralytics opencv-python
fi

# 6. Run the main server
echo ">>> Setup complete. Starting the Main Server..."
python3 main_server.py
