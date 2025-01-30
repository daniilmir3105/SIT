import torch


if __name__ == '__main__':
    # x = torch.randn(1, 3, 224, 224)
    print("CUDA доступна:", torch.cuda.is_available())
    print("Устройство:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")