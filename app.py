import os
import cv2
import json
import time
import base64
import gc
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
# Allow CORS for all domains and routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Try to import Ultralytics YOLO (PyTorch) for local execution
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

# Path to YOLOv8 ONNX model (fallback for lightweight production deployment)
MODEL_PATH = 'yolov8s.onnx'

# Load model
net = None
yolo_model = None

if HAS_ULTRALYTICS:
    if os.path.exists('yolov8s.pt'):
        print("Loading YOLOv8 Small model (yolov8s.pt) using Ultralytics...")
        yolo_model = YOLO('yolov8s.pt')
    else:
        print("WARNING: yolov8s.pt not found for Ultralytics.")

# Fallback to ONNX if Ultralytics is not available or failed to load a .pt model
if yolo_model is None:
    if os.path.exists(MODEL_PATH):
        print("Loading YOLOv8 ONNX model using OpenCV DNN...")
        net = cv2.dnn.readNetFromONNX(MODEL_PATH)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("ONNX Model loaded successfully.")
    else:
        print(f"WARNING: {MODEL_PATH} not found.")

# Global variables for caching latest stats
current_stats = {
    'total_vehicles': 0,
    'green_time': 10,
    'breakdown': {'Car': 0, 'Motorcycle': 0, 'Bus': 0, 'Truck': 0}
}

# Default YOLOv8 class mappings (COCO dataset indices)
# 0: Person (for camera testing/verification), 2: Car, 3: Motorcycle, 5: Bus, 7: Truck
class_names = {0: 'Person', 2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
colors = {0: (255, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0), 5: (0, 0, 255), 7: (0, 255, 255)}
predict_classes = [0, 2, 3, 5, 7]

if yolo_model is not None:
    # Dynamically extract classes from the loaded YOLO model names
    model_names = yolo_model.names
    print(f"\n[AI Setup] Loaded model classes from file: {model_names}")
    
    dynamic_class_names = {}
    dynamic_predict_classes = []
    
    # Target map for case-insensitive keyword matching
    target_map = {
        'person': 'Person',
        'car': 'Car',
        'motorcycle': 'Motorcycle',
        'motorbik': 'Motorcycle',
        'bike': 'Motorcycle',
        'bus': 'Bus',
        'truck': 'Truck',
        'ambulance': 'Car'  # Map ambulance to Car for traffic count
    }
    
    for idx, name in model_names.items():
        name_lower = name.lower()
        matched = False
        for target, standard_name in target_map.items():
            if target in name_lower:
                dynamic_class_names[idx] = standard_name
                dynamic_predict_classes.append(idx)
                matched = True
                break
        
        if not matched:
            if 'vehicle' in name_lower:
                dynamic_class_names[idx] = 'Car'
                dynamic_predict_classes.append(idx)
                
    if dynamic_predict_classes:
        class_names = dynamic_class_names
        predict_classes = dynamic_predict_classes
        print(f"[AI Setup] Mapped prediction classes: {class_names}")
        print(f"[AI Setup] Predicting on class indices: {predict_classes}\n")
    else:
        print("[AI Setup] WARNING: No target vehicle keywords matched. Predicting all classes.")
        class_names = {idx: name.capitalize() for idx, name in model_names.items()}
        predict_classes = list(model_names.keys())
        print(f"[AI Setup] Mapped classes: {class_names}\n")

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

# Add explicit CORS headers to every response
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/')
def index():
    return jsonify({
        "status": "AI Traffic API is running",
        "model_loaded": (net is not None) or (yolo_model is not None),
        "engine": "ultralytics" if yolo_model is not None else "opencv_dnn"
    })

@app.route('/data')
def get_data():
    return jsonify(current_stats)

@app.route('/process_frame', methods=['POST', 'OPTIONS'])
def process_frame():
    global current_stats, net, yolo_model
    
    # Handle preflight CORS request
    if request.method == 'OPTIONS':
        return jsonify({"status": "CORS preflight ok"}), 200
        
    try:
        # Check if any model is loaded
        if yolo_model is None and net is None:
            # Try reloading ONNX in case it was uploaded later
            if os.path.exists(MODEL_PATH):
                net = cv2.dnn.readNetFromONNX(MODEL_PATH)
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            else:
                return jsonify({"error": "No YOLO model loaded on server"}), 500

        if 'image' not in request.files:
            return jsonify({"error": "No image file provided in the request"}), 400
            
        file = request.files['image']
        img_bytes = file.read()
        
        # Convert bytes to numpy array and decode to OpenCV format
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400
            
        counts = {'Car': 0, 'Motorcycle': 0, 'Bus': 0, 'Truck': 0}
        total_vehicles = 0
        
        detected_boxes = []
        img_h, img_w = frame.shape[:2]
        
        if yolo_model is not None:
            # --- Inference using Ultralytics YOLO ---
            # Set confidence threshold to 0.15 and imgsz=320 for ultra-fast CPU inference
            results = yolo_model.predict(source=frame, classes=predict_classes, conf=0.15, imgsz=320, verbose=False)
            
            # Count the vehicles (ignoring Person for traffic calculations)
            raw_boxes = results[0].boxes
            
            for box in raw_boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                if class_id in class_names:
                    v_type = class_names[class_id]
                    if v_type != 'Person':
                        counts[v_type] += 1
                        total_vehicles += 1
                    
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detected_boxes.append({
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1),
                        "class": v_type,
                        "confidence": round(conf, 2)
                    })
            
        else:
            # --- Inference using OpenCV DNN with ONNX ---
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward()
            
            # For YOLOv8, outputs shape is (1, 84, 8400) or similar
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
                
            if len(outputs.shape) == 3:
                output = outputs[0].transpose() # Shape: (8400, 84)
            else:
                output = outputs.transpose()
                
            boxes = []
            confidences = []
            class_ids = []
            
            x_factor = img_w / 640.0
            y_factor = img_h / 640.0
            
            for row in output:
                classes_scores = row[4:]
                class_id = int(np.argmax(classes_scores))
                confidence = float(classes_scores[class_id])
                
                if confidence > 0.15 and class_id in class_names:
                    xc, yc, w, h = row[0:4]
                    left = int((xc - w/2) * x_factor)
                    top = int((yc - h/2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    
                    boxes.append([left, top, width, height])
                    confidences.append(confidence)
                    class_ids.append(class_id)
            
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.15, 0.5)
                if len(indices) > 0:
                    for i in indices:
                        idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                        box = boxes[idx]
                        cid = class_ids[idx]
                        conf = confidences[idx]
                        
                        v_type = class_names[cid]
                        if v_type != 'Person':
                            counts[v_type] += 1
                            total_vehicles += 1
                        
                        x, y, w, h = box
                        detected_boxes.append({
                            "x1": round(float(x), 1),
                            "y1": round(float(y), 1),
                            "x2": round(float(x + w), 1),
                            "y2": round(float(y + h), 1),
                            "class": v_type,
                            "confidence": round(float(conf), 2)
                        })
                
            del outputs
            gc.collect()
            
        green_time = get_green_light_time(total_vehicles)
        
        # Update global cache
        current_stats['total_vehicles'] = total_vehicles
        current_stats['green_time'] = green_time
        current_stats['breakdown'] = counts
        
        return jsonify({
            "total_vehicles": total_vehicles,
            "green_time": green_time,
            "breakdown": counts,
            "boxes": detected_boxes,
            "img_width": img_w,
            "img_height": img_h
        })
    except Exception as e:
        print(f"Error in process_frame: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
