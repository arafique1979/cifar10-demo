import torchvision.models as models
import torch.nn as nn

def build_vit(num_classes=10):
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model

