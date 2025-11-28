from models.simple_cnn import SimpleCNN
from models.resnet_model import build_resnet
from models.convnext_model import build_convnext
from models.vit_model import build_vit
from models.unet_model import UNet2D
from models.vit_tiny_model import ViTTinyPatch4
from models.deit_tiny_model import DeiTTiny

MODEL_REGISTRY = {
    "simple_cnn": lambda: SimpleCNN(),
    "resnet": lambda: build_resnet(),
    "convnext": lambda: build_convnext(),
    "vit": lambda: build_vit(),
    "unet": lambda: UNet2D(),
    "vit_tiny": lambda: ViTTinyPatch4(),
    "deit_tiny": lambda: DeiTTiny(),
}


def get_model(name):
    name = name.lower().strip()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}\nAvailable: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]()

