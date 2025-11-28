import torchvision.models as models
import torch.nn as nn

def build_resnet(num_classes=10):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
