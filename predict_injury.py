import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image

classes = [
    "bleeding",
    "burn",
    "fracture",
    "head_injury"
]

model = resnet18()
model.fc = nn.Linear(
    model.fc.in_features,
    4
)

model.load_state_dict(
    torch.load("injury_classifier.pth")
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

img = Image.open("test.jpg").convert("RGB")

img = transform(img).unsqueeze(0)

with torch.no_grad():

    output = model(img)

    pred = output.argmax(1)

print(
    "Predicted Injury:",
    classes[pred.item()]
)