import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from models.simple_cnn import SimpleCNN
from models.resnet_model import build_resnet
from models.convnext_model import build_convnext
from models.vit_model import build_vit
from models.unet_model import UNet2D


from utils.logger import get_logger

from tqdm import tqdm
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


from models.simple_cnn import SimpleCNN    # ADD THIS IMPORT


from models.registry import get_model

def load_model(name):
    return get_model(name)


def train_one_epoch(model, loader, criterion, optimizer, writer, epoch):
    model.train()
    correct = 0
    total = 0
    running_loss = 0

    loop = tqdm(loader, desc=f"Epoch {epoch} [train]")

    for i, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = pred.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        loop.set_postfix(loss=running_loss/(i+1), acc=100*correct/total)

        writer.add_scalar("Train/Loss", running_loss/(i+1), epoch*len(loader)+i)
        writer.add_scalar("Train/Accuracy", 100*correct/total, epoch*len(loader)+i)

def evaluate(model, loader, criterion, writer, epoch):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = criterion(pred, y)
            loss_sum += loss.item()

            _, predicted = pred.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    acc = 100 * correct / total
    loss_avg = loss_sum / len(loader)

    writer.add_scalar("Val/Loss", loss_avg, epoch)
    writer.add_scalar("Val/Accuracy", acc, epoch)

    print(f"\nValidation: Loss={loss_avg:.4f} | Accuracy={acc:.2f}%")

    return acc

def main(config_path):
    cfg = load_config(config_path)
    model_name = cfg["model_name"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    save_path = cfg["save_path"]

    print(f"\n🔥 Using model: {model_name}")
    print(f"⚙ Config: {cfg}\n")
    
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010))
    ])


    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    model = load_model(model_name).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load checkpoint if exists
    if os.path.exists(save_path):
        print(f"📥 Loading checkpoint from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))

    writer = get_logger(model_name)

    for epoch in range(1, epochs+1):
        train_one_epoch(model, trainloader, criterion, optimizer, writer, epoch)
        evaluate(model, testloader, criterion, writer, epoch)

        print(f"💾 Saving checkpoint → {save_path}")
        torch.save(model.state_dict(), save_path)

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/resnet.yaml")
    args = parser.parse_args()

    main(args.config)
