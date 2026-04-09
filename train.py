import torch
from torch.utils.data import DataLoader
from dataset import CropDataset
from models import get_model

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

# Load dataset
dataset = CropDataset("dataset/")

# 🔥 Reduce dataset for faster testing (REMOVE later for full training)
# dataset.data = dataset.data[:5000]

# DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = get_model(len(dataset.classes)).to(device)

# Optimizer + Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

# Training
epochs = 3   # 🔥 increase later (5–10 for final)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    print(f"\n🔥 Starting Epoch {epoch+1}/{epochs}")

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 🔥 Print progress every 20 batches
        if i % 20 == 0:
            print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"✅ Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), "model.pth")
print("\n💾 Model saved as model.pth")