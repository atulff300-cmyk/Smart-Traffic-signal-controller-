import cv2
from ultralytics import YOLO

# Pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')

image_path = 'check img trafific 2.jpg'
print("Original Model load ho gaya, image par check kar rahe hain...\n")

# classes=[2, 3, 5, 7] (Car, Motorcycle, Bus, Truck)
results = model.predict(source=image_path, classes=[2, 3, 5, 7], save=True, show=True)

# Kaunse number ka matlab kaunsi gaadi hai (COCO dataset ke hisaab se)
class_names = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

# Shuruat mein sabka count 0 rakhte hain
counts = {'Car': 0, 'Motorcycle': 0, 'Bus': 0, 'Truck': 0}

# Har ek box ko padho jo detect hua hai
for box in results[0].boxes:
    class_id = int(box.cls[0]) # Class ka number milega (jaise 2, 3 etc)
    if class_id in class_names:
        vehicle_type = class_names[class_id]
        counts[vehicle_type] += 1 # Jo gaadi mili, uska count +1 kar do

total_vehicles = sum(counts.values())

print(f"\n=====================================")
print(f"Total Vehicles (Kul Gaadiyan): {total_vehicles}")
print(f"=====================================")
print("Alag-alag Gaadiyon ka hisaab (Breakdown):")
for vehicle, count in counts.items():
    if count > 0: # Agar wo gaadi 0 se zyada hai tabhi print karo
        print(f"- {vehicle}: {count}")
print(f"=====================================\n")

# Screen rokne ke liye
cv2.waitKey(0)
cv2.destroyAllWindows()
