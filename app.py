import os
import cv2
import json
import time
import threading
import gc
import numpy as np
from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import torch

# Optimize PyTorch memory usage for 512MB RAM Limit on Render
torch.set_num_threads(1)

app = Flask(__name__)
CORS(app)  # Allow React app to fetch data
model = YOLO('yolov8n.pt')

# Global variables for single-thread frame generation & metrics caching
latest_frame_bytes = None
frame_lock = threading.Lock()

current_stats = {
    'total_vehicles': 0,
    'green_time': 10,
    'breakdown': {'Car': 0, 'Motorcycle': 0, 'Bus': 0, 'Truck': 0}
}

class_names = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

def get_green_light_time(vehicle_count):
    if vehicle_count == 0:
        return 10
    elif vehicle_count <= 5:
        return 20
    elif vehicle_count <= 10:
        return 30
    elif vehicle_count <= 20:
        return 45
    else:
        return 60

def yolo_processing_loop():
    global latest_frame_bytes, current_stats
    
    # Try opening webcam (0).
    cap = cv2.VideoCapture(0)
    use_fallback = not cap.isOpened()
    
    if use_fallback:
        print("Webcam not available. Running in Demo Simulation Mode.")
        sample_img = cv2.imread('check img trafific 2.jpg')
        if sample_img is None:
            # Create a blank fallback screen if sample image is missing
            sample_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(sample_img, "Camera Offline - Demo Mode", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        print("Webcam successfully initialized.")
        
    while True:
        try:
            if use_fallback:
                # 5 FPS in fallback/demo mode to save CPU
                time.sleep(0.2)
                frame = sample_img.copy()
                
                # Add live timestamp overlay
                h, w, _ = frame.shape
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                cv2.putText(frame, f"LIVE DEMO - {timestamp}", (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                success, frame = cap.read()
                if not success:
                    time.sleep(0.05)
                    continue
            
            # Run YOLO in inference mode (no gradients computed, saves memory)
            with torch.no_grad():
                results = model.predict(source=frame, classes=[2, 3, 5, 7], verbose=False)
            
            counts = {'Car': 0, 'Motorcycle': 0, 'Bus': 0, 'Truck': 0}
            total_vehicles = 0
            
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                if class_id in class_names:
                    v_type = class_names[class_id]
                    counts[v_type] += 1
                    total_vehicles += 1
                    
            green_time = get_green_light_time(total_vehicles)
            annotated_frame = results[0].plot()
            
            if use_fallback:
                cv2.putText(annotated_frame, "DEMO STREAM: Webcam Offline", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
            # Compress and encode to jpeg
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                with frame_lock:
                    latest_frame_bytes = frame_bytes
                    current_stats['total_vehicles'] = total_vehicles
                    current_stats['green_time'] = green_time
                    current_stats['breakdown'] = counts
            
            # Explicitly delete results and trigger garbage collection to free memory
            del results
            gc.collect()
                    
        except Exception as e:
            print(f"Error in YOLO thread: {e}")
            time.sleep(1)

# Start single-processing thread immediately
processing_thread = threading.Thread(target=yolo_processing_loop, daemon=True)
processing_thread.start()

def generate_frames():
    global latest_frame_bytes
    while True:
        # Stream at ~15 FPS to clients
        time.sleep(0.066)
        with frame_lock:
            if latest_frame_bytes is None:
                continue
            frame_bytes = latest_frame_bytes
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return jsonify({"status": "AI Traffic API is running"})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def get_data():
    return jsonify(current_stats)

if __name__ == '__main__':
    # Dynamically bind to PORT assigned by Render, default to 5000 locally
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


