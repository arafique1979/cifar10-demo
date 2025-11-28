import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchinfo import summary

from models.simple_cnn import SimpleCNN
from models.resnet_model import build_resnet
from models.convnext_model import build_convnext
from models.vit_model import build_vit

from utils.logger import get_logger
from utils.augment import build_transforms, cutmix, mixup
from utils.metrics import plot_confusion_matrix
from utils.scheduler import build_scheduler
from utils.metrics_saver import MetricSaver
from utils.metrics_plot import plot_curves

import argparse
import time

# =========================================================
# CPU-only optimized mode
# =========================================================
DEVICE = "cpu"
torch.set_num_threads(os.cpu_count())
torch.backends.opt_einsum.enabled = True

print(f"🧠 CPU-optimized training on {os.cpu_count()} threads")


CLASSES = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']



# =========================================================
# Load config
# =========================================================
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)



# =========================================================
# Model loader
# =========================================================
def load_model(name):
    if name == "resnet":
        return build_resnet()
    elif name == "convnext":
        return build_convnext()
    elif name == "vit":
        return build_vit()
    elif name == "simple_cnn":
        return SimpleCNN()
    else:
        raise ValueError(f"Unknown model: {name}")



# =========================================================
# Training loop (CPU optimized)
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, writer, epoch, epochs, scheduler):
    model.train()
    correct = 0
    total = 0
    running_loss = 0

    loop = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [train]", dynamic_ncols=True)

    start_time = time.time()

    for i, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)

        # CutMix / Mixup (lightweight)
        if torch.rand(1).item() < 0.5:
            if torch.rand(1).item() < 0.5:
                x, y1, y2, lam = cutmix(x, y, device=DEVICE)
            else:
                x, y1, y2, lam = mixup(x, y, device=DEVICE)
            mixed = True
        else:
            mixed = False

        optimizer.zero_grad()

        # Forward
        pred = model(x)

        # Loss
        if mixed:
            loss = lam * criterion(pred, y1) + (1 - lam) * criterion(pred, y2)
        else:
            loss = criterion(pred, y)

        # Backward + optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Stats
        running_loss += loss.item()
        _, predicted = pred.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        avg_loss = running_loss / (i + 1)
        avg_acc = 100 * correct / total

        # Update live display
        loop.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "acc": f"{avg_acc:.2f}%",
            "lr": f"{scheduler.get_last_lr()[0]:.5f}"
        })

        # TensorBoard
        step = epoch * len(loader) + i
        writer.add_scalar("Train/Loss", avg_loss, step)
        writer.add_scalar("Train/Accuracy", avg_acc, step)

    scheduler.step()
    epoch_time = time.time() - start_time
    return avg_loss, avg_acc, epoch_time



# =========================================================
# Validation loop
# =========================================================
def evaluate(model, loader, criterion, writer, epoch):
    model.eval()
    loss_sum = 0
    correct = 0
    total = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = criterion(pred, y)

            loss_sum += loss.item()

            _, predicted = pred.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            all_true.extend(y.cpu().tolist())
            all_pred.extend(predicted.cpu().tolist())

    val_loss = loss_sum / len(loader)
    val_acc = 100 * correct / total

    # Print to console
    print(f"\n📊 Validation: Loss={val_loss:.4f} | Acc={val_acc:.2f}%\n")

    # TensorBoard
    writer.add_scalar("Val/Loss", val_loss, epoch)
    writer.add_scalar("Val/Accuracy", val_acc, epoch)

    # Confusion matrix PNG
    plot_confusion_matrix(all_true, all_pred, CLASSES, f"runs/cm_epoch_{epoch}.png")

    return val_loss, val_acc



# =========================================================
# Main training entry
# =========================================================
def main(config_path):

    cfg = load_config(config_path)
    model_name = cfg["model_name"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    save_path = cfg["save_path"]

    print(f"\n🔥 Training model: {model_name}")
    print(f"⚙ Config: {cfg}")

    # Transforms
    transform = build_transforms()

    # Data
    trainset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=False)

    # Model
    model = load_model(model_name).to(DEVICE)

    print(summary(model, input_size=(1, 3, 32, 32)))

    # Try torch.compile()
    try:
        model = torch.compile(model)
        print("⚡ Using torch.compile() for CPU speed boost")
    except Exception as e:
        print(f"⚠️ compile() unavailable: {e}")

    # Loss / optimizer / scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = build_scheduler(optimizer, warmup_epochs=2, total_epochs=epochs)

    # Logging
    writer = get_logger(model_name)
    metric_saver = MetricSaver(f"runs/{model_name}_metrics.csv")

    train_losses, train_accs = [], []
    val_losses, val_accs = []

    best_acc = 0

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, epoch_time = train_one_epoch(
            model, trainloader, criterion, optimizer, writer, epoch, epochs, scheduler
        )

        val_loss, val_acc = evaluate(model, testloader, criterion, writer, epoch)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Save CSV
        metric_saver.log(epoch, train_loss, train_acc, val_loss, val_acc)

        # Best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"💾 Saving BEST model → {save_path}")
            torch.save(model.state_dict(), save_path)

        print(f"⏱ Epoch time: {epoch_time:.2f} sec\n")

    # Save curves PNG
    plot_curves(train_losses, train_accs, val_losses, val_accs, f"runs/{model_name}_curves.png")
    print(f"📈 Saved training curves → runs/{model_name}_curves.png")

    writer.close()



# =========================================================
# Run launcher
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/simple_cnn.yaml")
    args = parser.parse_args()
    main(args.config)

