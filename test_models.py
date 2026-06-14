from ultralytics import YOLO

fracture_model = YOLO("runs/detect/train/weights/best.pt")
burn_model = YOLO("runs/classify/train-5/weights/best.pt")

print("Fracture model loaded successfully!")
print("Burn model loaded successfully!")