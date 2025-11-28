import csv, os

class MetricSaver:
    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch","train_loss","train_acc","val_loss","val_acc"]
            )
    def log(self, epoch, train_loss, train_acc, val_loss, val_acc):
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, train_loss, train_acc, val_loss, val_acc]
            )
