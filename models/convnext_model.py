import torchvision.models as models
import torch.nn as nn

def build_convnext(num_classes=10):
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model
