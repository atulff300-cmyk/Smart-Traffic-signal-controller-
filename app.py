import os
import cv2
import json
import time
import threading
import gc
import random
import numpy as np
from flask import Flask, Response, jsonify
from flask_cors import CORS

# Check if running on Render cloud or if forced simulation is requested
IS_RENDER = 'RENDER' in os.environ or os.environ.get('USE_SIMULATION') == 'true'

# Conditional loading of heavy ML libraries to keep memory under 50MB on Render
HAS_YOLO = False
if not IS_RENDER:
    try:
        from ultralytics import YOLO
        import torch
        # Optimize PyTorch memory usage for local CPU runs
        torch.set_num_threads(1)
        model = YOLO('yolov8n.pt')
        HAS_YOLO = True
        print("YOLOv8 initialized successfully.")
    except ImportError:
        print("YOLO/Torch libraries not found. Falling back to Simulation Mode.")
else:
    print("Running on Render Cloud. Bypassing heavy YOLO/Torch imports to prevent RAM crashes.")

app = Flask(__name__)
CORS(app)  # Allow React app to fetch data

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
    
    if HAS_YOLO:
        # Real YOLO processing (for local development with Webcam)
        cap = cv2.VideoCapture(0)
        use_fallback = not cap.isOpened()
        
        if use_fallback:
            print("Webcam not available. Running real YOLO on fallback image.")
            sample_img = cv2.imread('check img trafific 2.jpg')
            if sample_img is None:
                sample_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(sample_img, "Camera Offline - Demo Mode", (120, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            print("Webcam successfully initialized.")
            
        while True:
            try:
                if use_fallback:
                    time.sleep(0.2)
                    frame = sample_img.copy()
                    h, w, _ = frame.shape
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    cv2.putText(frame, f"LIVE DEMO - {timestamp}", (20, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    success, frame = cap.read()
                    if not success:
                        time.sleep(0.05)
                        continue
                
                # Run YOLO in inference mode (saves memory)
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
                
                del results
                gc.collect()
                        
            except Exception as e:
                print(f"Error in YOLO thread: {e}")
                time.sleep(1)
    else:
        # Lightweight Simulation Mode (runs on Render under 50MB RAM)
        print("Starting lightweight traffic simulation loop.")
        sample_img = cv2.imread('check img trafific 2.jpg')
        if sample_img is None:
            sample_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(sample_img, "Camera Offline - Demo Mode", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
        # Define realistic bounding boxes on the static image
        # Format: (label, x1_ratio, y1_ratio, x2_ratio, y2_ratio, color)
        MOCK_VEHICLES = [
            ('Car', 0.15, 0.50, 0.35, 0.75, (0, 255, 0)),
            ('Car', 0.45, 0.55, 0.65, 0.85, (0, 255, 0)),
            ('Motorcycle', 0.70, 0.48, 0.80, 0.62, (255, 255, 0)),
            ('Bus', 0.05, 0.35, 0.38, 0.70, (0, 165, 255)),
            ('Truck', 0.32, 0.28, 0.58, 0.58, (255, 0, 0))
        ]
        
        last_change_time = 0
        active_vehicles = MOCK_VEHICLES[:2]  # Default starting vehicles
        
        while True:
            try:
                time.sleep(0.3)  # Loop at ~3 FPS to consume near 0% CPU
                current_time = time.time()
                
                # Shift simulated traffic every 6 seconds
                if current_time - last_change_time > 6:
                    num_active = random.randint(1, len(MOCK_VEHICLES))
                    active_vehicles = random.sample(MOCK_VEHICLES, num_active)
                    last_change_time = current_time
                
                frame = sample_img.copy()
                h, w, _ = frame.shape
                
                counts = {'Car': 0, 'Motorcycle': 0, 'Bus': 0, 'Truck': 0}
                total_vehicles = 0
                
                # Draw mock YOLO boxes and labels
                for label, rx1, ry1, rx2, ry2, color in active_vehicles:
                    counts[label] += 1
                    total_vehicles += 1
                    
                    x1, y1 = int(rx1 * w), int(ry1 * h)
                    x2, y2 = int(rx2 * w), int(ry2 * h)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    conf = random.randint(84, 98)
                    text = f"{label} {conf}%"
                    cv2.putText(frame, text, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                
                green_time = get_green_light_time(total_vehicles)
                
                # Overlay HUD text
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                cv2.putText(frame, f"LIVE DEMO - {timestamp}", (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "SIMULATOR: Webcam Offline", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    with frame_lock:
                        latest_frame_bytes = frame_bytes
                        current_stats['total_vehicles'] = total_vehicles
                        current_stats['green_time'] = green_time
                        current_stats['breakdown'] = counts
            except Exception as e:
                print(f"Error in Simulation loop: {e}")
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



