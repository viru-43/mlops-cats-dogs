import torch
import torch.nn as nn
from torchvision import models


def get_model():
    model = models.resnet18(weights="IMAGENET1K_V1")

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    return model
