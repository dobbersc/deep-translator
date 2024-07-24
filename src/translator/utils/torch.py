import torch


def detect_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
