import torch


def setup_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        device = "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS")
        device = "mps"
    else:
        print("Using CPU")
        device = "cpu"
    return device
