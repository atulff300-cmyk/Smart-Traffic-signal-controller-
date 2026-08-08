import os
import cv2
import json
import time
import threading
import gc
import base64
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import torch
from ultralytics import YOLO

app = Flask(__name__)
# Allow CORS for all domains and routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Load YOLO model
torch.set_num_threads(1)
model = YOLO('yolov8n.pt')

# Global variables for caching latest stats
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

# Add explicit CORS headers to every response (fixes preflight & error responses)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def index():
    return jsonify({"status": "AI Traffic API is running"})

@app.route('/data')
def get_data():
    return jsonify(current_stats)

@app.route('/process_frame', methods=['POST', 'OPTIONS'])
def process_frame():
    global current_stats
    
    # Handle preflight CORS request
    if request.method == 'OPTIONS':
        return jsonify({"status": "CORS preflight ok"}), 200
        
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided in the request"}), 400
            
        file = request.files['image']
        img_bytes = file.read()
        
        # Convert bytes to numpy array and decode to OpenCV format
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400
            
        # Run YOLOv8 prediction
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
        
        # Compress and encode to jpeg
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            return jsonify({"error": "Failed to encode annotated image"}), 500
            
        # Base64 encode the JPEG image to send to frontend
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Update global stats cache
        current_stats['total_vehicles'] = total_vehicles
        current_stats['green_time'] = green_time
        current_stats['breakdown'] = counts
        
        # Free up memory explicitly to prevent Render free-tier RAM crash
        del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return jsonify({
            "annotated_image": f"data:image/jpeg;base64,{img_base64}",
            "total_vehicles": total_vehicles,
            "green_time": green_time,
            "breakdown": counts
        })
    except Exception as e:
        print(f"Error in process_frame: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Dynamically bind to PORT assigned by Render, default to 5000 locally
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
