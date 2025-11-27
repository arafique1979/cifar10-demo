from torch.utils.tensorboard import SummaryWriter

def get_logger(run_name):
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    return writer
