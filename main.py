# -*- coding: utf-8 -*-
# from util.My_tool1 import *
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn

from pytorch_msssim import ssim

# -------------------------------
# Определение архитектуры модели SIT (минимальный вариант)
# -------------------------------
class HeadBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(HeadBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5_1 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        self.conv5x5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x = self.activation(self.conv3x3(x))
        x1 = self.activation(self.conv5x5_1(x)) + x
        x2 = self.activation(self.conv5x5_2(x1)) + x1
        return x2

class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, nhead=8, ff_dim=256):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, d_model)
        )
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        return x

def flatten_features(x):
    B, C, H, W = x.size()
    return x.view(B, C, H * W).permute(0, 2, 1)

def unflatten_features(x, H, W):
    B, N, C = x.size()
    return x.permute(0, 2, 1).view(B, C, H, W)

class EncoderTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=8, ff_dim=256):
        super(EncoderTransformer, self).__init__()
        self.transformer = TransformerBlock(d_model=d_model, nhead=nhead, ff_dim=ff_dim)
    def forward(self, x):
        B, C, H, W = x.size()
        x_seq = flatten_features(x)
        x_seq = self.transformer(x_seq)
        return unflatten_features(x_seq, H, W)

class USTBlock(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=128, nhead=8, ff_dim=512):
        super(USTBlock, self).__init__()
        self.down = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        self.transformer = TransformerBlock(d_model=hidden_channels, nhead=nhead, ff_dim=ff_dim)
        self.up = nn.ConvTranspose2d(hidden_channels, in_channels, kernel_size=2, stride=2)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        residual = x
        x_down = self.activation(self.down(x))
        B, C, H, W = x_down.size()
        x_seq = flatten_features(x_down)
        x_seq = self.transformer(x_seq)
        x_trans = unflatten_features(x_seq, H, W)
        x_up = self.activation(self.up(x_trans))
        return x_up + residual

class DecoderTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=8, ff_dim=256):
        super(DecoderTransformer, self).__init__()
        self.transformer = TransformerBlock(d_model=d_model, nhead=nhead, ff_dim=ff_dim)
    def forward(self, x, skip):
        B, C, H, W = x.size()
        x = x + skip
        x_seq = flatten_features(x)
        x_seq = self.transformer(x_seq)
        return unflatten_features(x_seq, H, W)

class TailBlock(nn.Module):
    def __init__(self, in_channels=64):
        super(TailBlock, self).__init__()
        self.down = nn.Conv2d(in_channels, in_channels*2, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels*2, in_channels*2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels*2, in_channels*2, kernel_size=3, padding=1)
        self.up = nn.ConvTranspose2d(in_channels*2, in_channels, kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=1)
        self.final_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    def forward(self, x):
        x_down = F.relu(self.down(x))
        x_attn = F.relu(self.conv1(x_down))
        x_attn = F.relu(self.conv2(x_attn))
        x_up = F.relu(self.up(x_attn))
        attn_weights = self.softmax(x_up)
        return self.final_conv(x * attn_weights)

class SITModel(nn.Module):
    def __init__(self):
        super(SITModel, self).__init__()
        self.head = HeadBlock(in_channels=1, out_channels=64)
        self.encoder = EncoderTransformer(d_model=64, nhead=8, ff_dim=256)
        self.ust1 = USTBlock(in_channels=64, hidden_channels=128, nhead=8, ff_dim=512)
        self.ust2 = USTBlock(in_channels=64, hidden_channels=128, nhead=8, ff_dim=512)
        self.decoder = DecoderTransformer(d_model=64, nhead=8, ff_dim=256)
        self.tail = TailBlock(in_channels=64)
    def forward(self, x):
        x_head = self.head(x)
        x_enc = self.encoder(x_head)
        x_ust = self.ust1(x_enc)
        x_ust = self.ust2(x_ust)
        x_dec = self.decoder(x_ust, x_enc)
        return self.tail(x_dec)

# -------------------------------
# Функция потерь L1 + SSIM
# -------------------------------
def l1_ssim_loss(pred, target, alpha=1.0, beta=1.0):
    l1_loss = F.l1_loss(pred, target)
    ssim_val = ssim(pred, target, data_range=target.max() - target.min(), size_average=True)
    return alpha * l1_loss + beta * (1 - ssim_val)

# -------------------------------
# Функция предварительной обработки сейсмических данных
# -------------------------------
def preprocess_seismic_data(seismic_data, nsub, noise_level=0.1):
    """
    Добавляет гауссов шум и прореживает трассы:
      - Из каждой nsub-той трассы оставляет данные, остальные зануляет.
    """
    noise = np.random.normal(0, noise_level * np.std(seismic_data), seismic_data.shape)
    noisy_data = seismic_data + noise
    kept_traces = np.arange(0, noisy_data.shape[1], nsub)
    subsampled_data = noisy_data.copy()
    for trace_idx in range(subsampled_data.shape[1]):
        if trace_idx not in kept_traces:
            subsampled_data[:, trace_idx] = 0
    return subsampled_data

# -------------------------------
# Функция инференса по патчам для полного изображения
# -------------------------------
def inference_full_data(model, data, patch_size=64, stride=64):
    """
    Разбивает входное изображение data (H, W) на неперекрывающиеся патчи (с указанным stride),
    прогоняет каждый патч через модель и объединяет результаты.
    """
    H, W = data.shape
    output = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = data[i:i+patch_size, j:j+patch_size]
            input_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float()
            with torch.no_grad():
                patch_out = model(input_tensor).squeeze().cpu().numpy()
            output[i:i+patch_size, j:j+patch_size] += patch_out
            count[i:i+patch_size, j:j+patch_size] += 1
    # На случай, если патчи не покрывают полностью изображение (можно доработать)
    output = output / (count + 1e-8)
    return output

# -------------------------------
# Эксперимент: загрузка модели и интерполяция данных
# -------------------------------
def run_experiment():
    # Загрузка оригинальных данных (файл содержит np.ndarray, транспонируем, чтобы строки – время, столбцы – трассы)
    original_data = np.load(r'C:\Users\Daniil\PycharmProjects\SIT\test_data\array_0.npy').T.astype(np.float64)
    original_data = original_data[21:, :64]
    np.save(r'C:\Users\Daniil\PycharmProjects\SIT\test_data\orig.npy', original_data)

    # Параметры предварительной обработки
    nsub = 2           # Из каждой группы из 2 трасс оставляем 1
    noise_level = 0    # Здесь задаём уровень шума

    # Получаем прорежённые данные
    subsampled_data = preprocess_seismic_data(original_data, nsub, noise_level)
    np.save(r'C:\Users\Daniil\PycharmProjects\SIT\test_data\new_noise_and_miss.npy', subsampled_data)

    # Загрузка обученной модели
    model = SITModel()
    model.load_state_dict(torch.load(r"C:\Users\Daniil\PycharmProjects\SIT\models\SIT_model.pth", map_location='cpu'))
    model.eval()

    # Если данные слишком большие, проводим инференс по патчам
    output_data = inference_full_data(model, subsampled_data, patch_size=64, stride=64)
    np.save(r'C:\Users\Daniil\PycharmProjects\SIT\test_data\USWIN_result.npy', output_data)

    # Вычисляем разницу между оригинальными и прорежёнными данными
    difference_data = original_data - subsampled_data

    # -------------------------------
    # Вывод 4 графиков
    # -------------------------------
    # График 1: Исходные данные
    plt.figure()
    plt.imshow(original_data, cmap='gray', aspect='auto', vmin=-0.2, vmax=0.2)
    plt.title("Исходные данные")
    plt.xlabel("Расстояние от источника, м")
    plt.ylabel("Время свободного пробега, мс")
    plt.show()

    # График 2: Прорежённые данные
    plt.figure()
    plt.imshow(subsampled_data, cmap='gray', aspect='auto', vmin=-0.2, vmax=0.2)
    plt.title("Прорежённые данные")
    plt.xlabel("Расстояние от источника, м")
    plt.ylabel("Время свободного пробега, мс")
    plt.show()

    # График 3: Результат работы нейросети (интерполированные данные)
    plt.figure()
    plt.imshow(output_data, cmap='gray', aspect='auto', vmin=-0.2, vmax=0.2)
    plt.title("Результат работы нейросети")
    plt.xlabel("Расстояние от источника, м")
    plt.ylabel("Время свободного пробега, мс")
    plt.show()

    # График 4: Разница между оригинальными и прорежёнными данными
    plt.figure()
    plt.imshow(original_data - output_data, cmap='gray', aspect='auto', vmin=-0.2, vmax=0.2)
    plt.title("Разница между оригинальными и прорежёнными данными")
    plt.xlabel("Расстояние от источника, м")
    plt.ylabel("Время свободного пробега, мс")
    plt.show()

    mse = np.mean((original_data - output_data) ** 2)
    print("MSE between original and denoised data: ", mse)
    np.save(r'C:\Users\Daniil\PycharmProjects\SIT\test_data\restored.npy', output_data)

if __name__ == '__main__':
    run_experiment()
