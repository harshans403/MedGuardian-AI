from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model(
    r"Dataset\fracture\bone fracture detection.v4-v4.yolov8\test\images"
)

for r in results:
    print("Detected:", len(r.boxes))