import os
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timm import create_model
from safetensors.torch import load_file


class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, output_size=(800, 800)):
        super(UNetDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        self.output_size = output_size

    def forward(self, x):
        x = self.decoder(x)
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        return x


class CombinedModel(nn.Module):
    def __init__(self, pretrained=False, swin_model_name='swin_tiny_patch4_window7_224', output_size=(800, 800)):
        super(CombinedModel, self).__init__()
        self.encoder = create_model(swin_model_name, pretrained=False)
        self.encoder.head = nn.Identity()
        self.decoder = UNetDecoder(in_channels=768, out_channels=1, output_size=output_size)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        enc_features = self.encoder.forward_features(x)
        if isinstance(enc_features, list):
            enc_features = enc_features[-1]
        if len(enc_features.shape) == 3:
            B, N, C = enc_features.shape
            H = W = int(N**0.5)
            enc_features = enc_features.permute(0, 2, 1).reshape(B, C, H, W)
        elif len(enc_features.shape) == 4:
            enc_features = enc_features.permute(0, 3, 1, 2).contiguous()
        output = self.decoder(enc_features)
        return output


def load_model(model_path, swin_model_name='swin_tiny_patch4_window7_224', output_size=(800, 800)):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Проверяем наличие файла модели
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели не найден по пути: {model_path}. Убедитесь, что он существует.")

    # Создаем модель
    model = CombinedModel(pretrained=False, swin_model_name=swin_model_name, output_size=output_size)

    # Загрузка весов
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_seismic_data(seismic_data, nsub, noise_level=0.1):
    noise = np.random.normal(0, noise_level * np.std(seismic_data), seismic_data.shape)
    noisy_data = seismic_data + noise
    kept_traces = np.arange(0, noisy_data.shape[1], nsub)
    subsampled_data = noisy_data.copy()
    for trace_idx in range(subsampled_data.shape[1]):
        if trace_idx not in kept_traces:
            subsampled_data[:, trace_idx] = 0
    return subsampled_data


def test_model(model, seismic_data, nsub):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Прореживание данных
    preprocessed_data = preprocess_seismic_data(seismic_data, nsub=nsub)

    # Преобразование данных для модели
    input_data = torch.tensor(preprocessed_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Восстановление данных
    with torch.no_grad():
        restored_data = model(input_data).squeeze().cpu().numpy()

    return preprocessed_data, restored_data


# def plot_results(original, subsampled, restored):
#     plt.figure(figsize=(10, 6))
#
#     plt.subplot(2, 2, 1)
#     plt.imshow(original, cmap='gray', aspect='auto', vmin=-1, vmax=1)
#     plt.title("Исходные данные")
#     plt.xlabel('Расстояние от источника, м')
#     plt.ylabel('Время свободного пробега, мс')
#
#     plt.subplot(2, 2, 2)
#     plt.imshow(subsampled, cmap='gray', aspect='auto', vmin=-1, vmax=1)
#     plt.title("Прореженные данные")
#     plt.xlabel('Расстояние от источника, м')
#     plt.ylabel('Время свободного пробега, мс')
#
#     plt.subplot(2, 2, 3)
#     plt.imshow(restored, cmap='gray', aspect='auto', vmin=-1, vmax=1)
#     plt.title("Результат работы нейросети")
#     plt.xlabel('Расстояние от источника, м')
#     plt.ylabel('Время свободного пробега, мс')
#
#     plt.subplot(2, 2, 4)
#     plt.imshow(original - restored, cmap='gray', aspect='auto', vmin=-1, vmax=1)
#     plt.title("Разница")
#     plt.xlabel('Расстояние от источника, м')
#     plt.ylabel('Время свободного пробега, мс')
#
#     plt.tight_layout()
#     plt.show()

def plot_results(original, subsampled, restored):
    # Приведение размеров restored к размерам original
    restored_resized = F.interpolate(
        torch.tensor(restored).unsqueeze(0).unsqueeze(0),
        size=original.shape,
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray', aspect='auto', vmin=-1, vmax=1)
    plt.title("Исходные данные")
    plt.xlabel('Расстояние от источника, м')
    plt.ylabel('Время свободного пробега, мс')

    plt.subplot(2, 2, 2)
    plt.imshow(subsampled, cmap='gray', aspect='auto', vmin=-1, vmax=1)
    plt.title("Прореженные данные")
    plt.xlabel('Расстояние от источника, м')
    plt.ylabel('Время свободного пробега, мс')

    plt.subplot(2, 2, 3)
    plt.imshow(restored_resized, cmap='gray', aspect='auto', vmin=-1, vmax=1)
    plt.title("Результат работы нейросети")
    plt.xlabel('Расстояние от источника, м')
    plt.ylabel('Время свободного пробега, мс')

    plt.subplot(2, 2, 4)
    plt.imshow(original - restored_resized, cmap='gray', aspect='auto', vmin=-1, vmax=1)
    plt.title(f"Разница, MSE={np.mean((original - restored_resized) ** 2)}")
    plt.xlabel('Расстояние от источника, м')
    plt.ylabel('Время свободного пробега, мс')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models", "combined_model_swin_tiny_patch4_window7_224.pth")
    # model_path = os.path.join(current_dir, "models", "combined_model_swin_tiny_patch4_window7_224_2.pth")
    # model_path = os.path.join(current_dir, "models", "combined_model_swin_tiny_patch4_window7_224_100.pth")

    # Загрузка обученной модели
    model = load_model(model_path)

    # Загрузка тестовых данных
    test_data_path = os.path.join(current_dir, "test_data", "data_dense.npy")
    # test_data_path = os.path.join(current_dir, "data", "part_1.npy")
    seismic_data = np.load(test_data_path)
    # seismic_data = (seismic_data - np.min(seismic_data)) / (np.max(seismic_data) - np.min(seismic_data) + 1e-6)


    # Прореживание и восстановление
    nsub = 2  # Прореживание каждых 2 трасс
    subsampled_data, restored_data = test_model(model, seismic_data, nsub)

    # Построение графиков
    plot_results(seismic_data, subsampled_data, restored_data)
