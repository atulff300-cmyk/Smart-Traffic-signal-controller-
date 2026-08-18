import os
from ultralytics import YOLO

def export_model():
    if os.path.exists('yolov8s.pt'):
        print("Found model 'yolov8s.pt'. Exporting to ONNX...")
        model = YOLO('yolov8s.pt')
        model.export(format='onnx', imgsz=640)
        print("Export complete. 'yolov8s.onnx' should be created.")
    elif os.path.exists('best.pt'):
        print("Found custom model 'best.pt'. Exporting to ONNX...")
        model = YOLO('best.pt')
        model.export(format='onnx', imgsz=640)
        print("Export complete. 'best.onnx' should be created.")
    elif os.path.exists('yolov8n.pt'):
        print("Found base model 'yolov8n.pt'. Exporting to ONNX...")
        model = YOLO('yolov8n.pt')
        model.export(format='onnx', imgsz=640)
        print("Export complete. 'yolov8n.onnx' should be created.")
    else:
        print("Error: No .pt model found to export!")

if __name__ == '__main__':
    export_model()
