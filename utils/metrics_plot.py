import matplotlib.pyplot as plt

def plot_curves(train_losses, train_accs, val_losses, val_accs, save_path):
    """
    Saves PNG file with two subplots:
      - Loss curve (train & val)
      - Accuracy curve (train & val)
    """

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # ---------- LOSS CURVE ----------
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs, val_losses, label="Val Loss", marker='o')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # ---------- ACCURACY CURVE ----------
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Accuracy", marker='o')
    plt.plot(epochs, val_accs, label="Val Accuracy", marker='o')
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
