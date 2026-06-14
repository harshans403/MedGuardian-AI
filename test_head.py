from ultralytics import YOLO

model = YOLO("runs/detect/train-6/weights/best.pt")

results = model(
    r"Dataset\head injury\bruises.v2i.yolov8\test\images\istockphoto-1352456481-612x612_jpg.rf.df6702d21e3e88378e771a4d5d32b0f0.jpg",
    save=True,
    conf=0.25
)

for r in results:
    print("Detected:", len(r.boxes))