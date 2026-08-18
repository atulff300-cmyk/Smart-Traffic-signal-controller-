from ultralytics import YOLO

# Model load kar rahe hain (yolov8s - small version jo fine accuracy deta hai)
model = YOLO('yolov8s.pt')

# Training shuru kar rahe hain
model.train(
    data='C:/Users/Atul/Downloads/Veichle-detection.yolov8 (1)/data.yaml',
    epochs=50, # 50 baar dataset ko dekhega seekhne ke liye
    imgsz=640, # Image size resize karke 640x640 pe train karega
    batch=16 # Ek baar mein 16 images process karega
)
