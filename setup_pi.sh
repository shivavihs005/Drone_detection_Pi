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
    python3-numpy \
    python3-opencv \
    python3-scipy \
    alsa-utils \
    libportaudio2 \
    libsndfile1 \
    v4l-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libatlas-base-dev \
    libopenblas-dev \
    ffmpeg

# 2. Configure I2S INMP441 Microphone
# Pin wiring:
#   VCC  -> 3.3V (Pin 1)
#   GND  -> GND  (Pin 6)
#   WS   -> GPIO19 (Pin 35)
#   SCK  -> GPIO18 (Pin 12)
#   SD   -> GPIO20 (Pin 38)
#   L/R  -> GND (left channel)
echo ">>> Configuring I2S microphone (INMP441)..."

# Determine config.txt location (Bookworm uses /boot/firmware/, older uses /boot/)
if [ -f /boot/firmware/config.txt ]; then
    BOOT_CONFIG="/boot/firmware/config.txt"
elif [ -f /boot/config.txt ]; then
    BOOT_CONFIG="/boot/config.txt"
else
    echo ">>> WARNING: Could not find boot config.txt"
    BOOT_CONFIG=""
fi

I2S_REBOOT_NEEDED=false

if [ -n "$BOOT_CONFIG" ]; then
    # Enable I2S
    if ! grep -q "^dtparam=i2s=on" "$BOOT_CONFIG"; then
        echo ">>> Enabling I2S in $BOOT_CONFIG..."
        echo "dtparam=i2s=on" | sudo tee -a "$BOOT_CONFIG"
        I2S_REBOOT_NEEDED=true
    else
        echo ">>> I2S already enabled in $BOOT_CONFIG."
    fi

    # Add I2S sound card overlay for INMP441
    if ! grep -q "^dtoverlay=googlevoicehat-soundcard" "$BOOT_CONFIG" && \
       ! grep -q "^dtoverlay=i2s-mmap" "$BOOT_CONFIG"; then
        echo ">>> Adding I2S mic overlay..."
        echo "dtoverlay=googlevoicehat-soundcard" | sudo tee -a "$BOOT_CONFIG"
        I2S_REBOOT_NEEDED=true
    else
        echo ">>> I2S overlay already configured."
    fi
fi

# Load I2S kernel modules (for current session, persistent after reboot via overlay)
sudo modprobe snd-bcm2835 2>/dev/null || true

# 3. Verify USB webcam is detected
echo ">>> Checking for USB webcam..."
if v4l2-ctl --list-devices 2>/dev/null | grep -q "video"; then
    echo ">>> USB webcam detected:"
    v4l2-ctl --list-devices
else
    echo ">>> WARNING: No USB webcam detected. Please connect one before running the server."
fi

# 4. Verify I2S microphone is detected
echo ">>> Checking for I2S microphone..."
if arecord -l 2>/dev/null | grep -q "card"; then
    echo ">>> Audio capture devices found:"
    arecord -l
else
    if [ "$I2S_REBOOT_NEEDED" = true ]; then
        echo ">>> I2S configuration added. A REBOOT is required for the mic to appear."
    else
        echo ">>> WARNING: No audio capture devices found. Check INMP441 wiring."
    fi
fi

# 5. Update the repository from GitHub
echo ">>> Pulling latest code from GitHub..."
git pull origin main

# 6. Create virtual environment
ENV_DIR="drone_env"
if [ ! -d "$ENV_DIR" ]; then
    echo ">>> Creating virtual environment in $ENV_DIR..."
    python3 -m venv --system-site-packages $ENV_DIR
else
    echo ">>> Virtual environment already exists."
fi

# 7. Activate virtual environment
echo ">>> Activating virtual environment..."
source $ENV_DIR/bin/activate

# 8. Install required Python packages
# On Pi ARM, opencv/numpy/scipy come from apt (system packages).
# The venv uses --system-site-packages so they're already available.
# PyPI x86 wheels cause "Illegal instruction" on ARM, so we must:
#   1. Install ultralytics WITHOUT deps
#   2. Manually install safe deps
#   3. Nuke ALL pip-installed ARM-incompatible packages
#   4. Verify every import individually
echo ">>> Installing Python dependencies..."
python3 -m pip install --upgrade pip

# Step A: Install ultralytics WITHOUT letting it pull in x86 torch/opencv/numpy
echo ">>> Installing ultralytics (no-deps to prevent x86 packages)..."
pip install --no-deps ultralytics

# Step B: Install pure-python / safe dependencies
echo ">>> Installing safe Python packages..."
pip install Flask sounddevice requests librosa joblib scikit-learn pyyaml pillow tqdm psutil py-cpuinfo matplotlib pandas seaborn

# Step C: Forcefully remove ALL pip-installed packages that crash on ARM
echo ">>> Purging x86-compiled packages that crash on ARM..."
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null || true
pip uninstall -y numpy 2>/dev/null || true
pip uninstall -y scipy 2>/dev/null || true
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Step D: Install PyTorch for ARM (aarch64) from PyTorch's official CPU index
echo ">>> Installing PyTorch for ARM..."
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    # PyTorch official CPU wheels include aarch64 builds
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    # If that installed x86 numpy, remove it again
    python3 -c "import numpy" 2>/dev/null || pip uninstall -y numpy 2>/dev/null || true
else
    pip install torch torchvision
fi

# Step E: Make sure numpy/opencv/scipy come from APT, not pip
# The venv was created with --system-site-packages, so apt packages are visible
echo ">>> Ensuring system (apt) packages are used for numpy/opencv/scipy..."
pip uninstall -y numpy 2>/dev/null || true
pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
pip uninstall -y scipy 2>/dev/null || true

# Step F: Verify EVERY import that main_server.py needs
echo ">>> Verifying all imports work on this ARM CPU..."
IMPORT_OK=true

for mod in numpy cv2 scipy torch torchvision flask sounddevice requests; do
    if python3 -c "import $mod; print('[OK] $mod')" 2>/dev/null; then
        :
    else
        echo "[FAIL] $mod crashed or not found!"
        IMPORT_OK=false
    fi
done

# Test ultralytics + YOLO specifically (this loads torch internally)
if python3 -c "from ultralytics import YOLO; print('[OK] ultralytics YOLO')" 2>/dev/null; then
    :
else
    echo "[FAIL] ultralytics YOLO import crashed!"
    IMPORT_OK=false
fi

# Test the actual server imports
if python3 -c "
from camera_module import CameraModule
from audio_module import AudioModule
from fusion_module import FusionModule
print('[OK] All server modules import successfully')
" 2>/dev/null; then
    :
else
    echo "[FAIL] Server module import crashed!"
    IMPORT_OK=false
fi

if [ "$IMPORT_OK" = false ]; then
    echo ""
    echo "=========================================="
    echo " IMPORT ERRORS DETECTED"
    echo "=========================================="
    echo " Some packages are not ARM-compatible."
    echo " Check the [FAIL] messages above."
    echo " You may need to install torch from source:"
    echo "   pip install torch --extra-index-url https://www.piwheels.org/simple"
    echo "=========================================="
    echo " Attempting to start server anyway..."
fi

# 9. Create required directories
echo ">>> Creating required directories..."
mkdir -p models
mkdir -p templates

# 10. Check if reboot needed
if [ "$I2S_REBOOT_NEEDED" = true ]; then
    echo ""
    echo "=========================================="
    echo " I2S CONFIG CHANGED - REBOOT REQUIRED!"
    echo "=========================================="
    echo " Run: sudo reboot"
    echo " Then re-run this script to start the server."
    echo "=========================================="
    exit 0
fi

# 11. Run the main server
echo ""
echo "=========================================="
echo " Setup complete!"
echo " USB webcam for video detection."
echo " I2S INMP441 mic for audio detection."
echo "=========================================="
echo ">>> Starting the Main Server..."
python3 main_server.py
