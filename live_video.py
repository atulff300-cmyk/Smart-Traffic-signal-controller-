import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

print("Live Video start ho rahi hai... (Band karne ke liye window par click karke 'q' dabayein)")
class_names = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

# Yeh function decide karega ki kitni gaadiyon par kitna time dena hai
def get_green_light_time(vehicle_count):
    if vehicle_count == 0:
        return 10  # Agar koi gaadi nahi hai toh 10 second
    elif vehicle_count <= 5:
        return 20  # 1 se 5 gaadiyan -> 20 second
    elif vehicle_count <= 10:
        return 30  # 6 se 10 gaadiyan -> 30 second
    elif vehicle_count <= 20:
        return 45  # 11 se 20 gaadiyan -> 45 second
    else:
        return 60  # 20 se zyada gaadiyan -> maximum 60 second

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(source=frame, classes=[2, 3, 5, 7], verbose=False)
    
    total_vehicles = 0
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if class_id in class_names:
            total_vehicles += 1
            
    # Gaadiyon ke count ke hisaab se Time nikalo
    green_time = get_green_light_time(total_vehicles)
    
    annotated_frame = results[0].plot()
    
    # Screen par Count (Total Vehicles) likho
    cv2.putText(annotated_frame, f'Vehicles: {total_vehicles}', (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
    # Screen par Time (Green Light Time) likho (Neeche ki line mein, Yellow color mein)
    cv2.putText(annotated_frame, f'Green Time: {green_time} sec', (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3) 

    cv2.imshow("Smart Traffic AI - Live", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
