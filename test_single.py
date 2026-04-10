import torch
from PIL import Image
import torchvision.transforms as transforms
from models import get_model
from dataset import CropDataset

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

# 🔥 Load dataset (to get class names)
dataset = CropDataset("dataset/")
classes = dataset.classes

# 🔥 Load model with correct number of classes
model = get_model(len(classes)).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 🔥 Load your test image
img_path = "test.jpg"   # 👈 put your image here
image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    output = model(image)
    probs = torch.softmax(output, dim=1)

    pred = torch.argmax(probs).item()
    confidence = torch.max(probs).item()

# Output
print("\n🌱 Predicted Class Index:", pred)
print("🌿 Disease Name:", classes[pred])
print("📊 Confidence:", round(confidence, 4))