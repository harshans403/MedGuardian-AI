from ultralytics import YOLO

model = YOLO("runs/classify/train-5/weights/best.pt")

results = model("test.jpg")

print(results[0].names[results[0].probs.top1])