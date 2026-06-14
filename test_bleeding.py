from ultralytics import YOLO

model = YOLO("runs/detect/train-7/weights/best.pt")

results = model(
    r"Dataset\Bleeding\Injury detection.v2i.yolov8\test\images",
    save=True,
    conf=0.25
)

for r in results:
    print("Detected:", len(r.boxes))