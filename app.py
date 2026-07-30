import cv2
import json
from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app) # Allow React app to fetch data
model = YOLO('yolov8n.pt')

# Global variables to store current traffic stats
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

def generate_frames():
    global current_stats
    
    # 0 for webcam. Update to 'video.mp4' for video file
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
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
        
        # Update global stats so the web page can fetch them
        current_stats['total_vehicles'] = total_vehicles
        current_stats['green_time'] = green_time
        current_stats['breakdown'] = counts
        
        # Draw boxes on the frame
        annotated_frame = results[0].plot()
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield frame for the MJPEG stream
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
    # This endpoint returns the latest counts and time to the dashboard as JSON
    return jsonify(current_stats)

if __name__ == '__main__':
    # use_reloader=False prevents YOLO from initializing twice
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
