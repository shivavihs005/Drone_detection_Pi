from flask import Flask, Response, render_template, jsonify, request
import time
import signal
import sys
from camera_module import CameraModule
from audio_module import AudioModule
from fusion_module import FusionModule
from api_client import api_client

app = Flask(__name__)

# Initialize singletons
camera = CameraModule()
audio = AudioModule()
fusion = FusionModule()

def init_system():
    print("[SYSTEM] Starting Camera Module thread...")
    camera.start()
    print("[SYSTEM] Starting Audio Module thread...")
    audio.start()
    print("[SYSTEM] System initialized successfully.")

def shutdown_handler(sig, frame):
    print("\n[SYSTEM] Received shutdown signal. Cleaning up threads...")
    camera.stop()
    audio.stop()
    print("[SYSTEM] Shutdown complete.")
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    while True:
        cam_data = camera.get_latest()
        frame_bytes = cam_data["frame"]
        
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.01)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/metrics')
def metrics():
    cam_data = camera.get_latest()
    aud_data = audio.get_latest()
    
    vis_conf = cam_data.get("confidence", 0.0)
    aud_conf = aud_data.get("confidence", 0.0)
    dom_freq = aud_data.get("dominant_freq", 0)
    
    fused = fusion.fuse(vis_conf, aud_conf)
    
    # Check threshold for detection locally
    drone_detected = fused["is_detected"]
    
    api_confirmed = False
    cloud_conf = 0.0
    
    if drone_detected:
        api_res = api_client.finalize_detection(cam_data.get("frame"), dom_freq, aud_conf)
        api_confirmed = api_res.get("drone_confirmed", False)
        cloud_conf = api_res.get("cloud_confidence", 0.0)
        
        # Optionally, stringently require API confirmation
        drone_detected = api_confirmed
    
    return jsonify({
        "vision_confidence": fused["vision"],
        "audio_confidence": fused["audio"],
        "fusion_confidence": fused["fusion"],
        "dominant_freq": dom_freq,
        "is_detected": drone_detected,
        "api_confirmed": api_confirmed,
        "cloud_confidence": cloud_conf,
        "vision_enabled": camera.vision_enabled,
        "audio_enabled": audio.audio_enabled
    })

@app.route('/api/toggle', methods=['POST'])
def toggle_sensor():
    data = request.json
    if "vision_enabled" in data:
        camera.vision_enabled = data["vision_enabled"]
        print(f"[SYSTEM] Vision Sensor Enabled: {camera.vision_enabled}")
    if "audio_enabled" in data:
        audio.audio_enabled = data["audio_enabled"]
        print(f"[SYSTEM] Audio Sensor Enabled: {audio.audio_enabled}")
    return jsonify({"status": "success"})

if __name__ == '__main__':
    init_system()
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
