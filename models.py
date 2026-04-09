import torch.nn as nn
import torchvision.models as models

def get_model(num_classes):
    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model  # ✅ make sure there's no extra parenthesis