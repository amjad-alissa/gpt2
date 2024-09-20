import torch


if torch.cuda.is_available():
    print("CUDA is available. Your GPU supports CUDA.")
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Your GPU may not support CUDA.")
