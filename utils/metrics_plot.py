import matplotlib.pyplot as plt

def plot_curves(train_losses, train_accs, val_losses, val_accs, save_path):
    epochs = range(1, len(train_losses)+1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.legend(); plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.legend(); plt.title("Accuracy Curve")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
