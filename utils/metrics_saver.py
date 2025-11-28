import csv
import os

class MetricSaver:
    """
    Saves epoch-level metrics (train/val) to a CSV file.
    Columns: epoch, train_loss, train_acc, val_loss, val_acc
    """

    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Initialize CSV with header
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Append a row of metrics to the CSV file."""
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
