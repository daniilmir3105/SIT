import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from safetensors.torch import load_file
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


class SeismicDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        if not self.files:
            raise ValueError(f"В директории {data_dir} нет файлов с расширением .npy.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        try:
            data = np.load(file_path)
            # Сохраняем параметры нормализации для денормализации
            data_min = np.min(data)
            data_max = np.max(data)
            data = (data - data_min) / (data_max - data_min + 1e-6)

            mask = np.ones_like(data)
            num_missing_traces = np.random.randint(5, 10)  # Уменьшил диапазон
            missing_indices = np.random.choice(
                data.shape[0],
                num_missing_traces,
                replace=False
            )
            mask[missing_indices, :] = 0
            corrupted_data = data * mask

            return {
                'corrupted': torch.tensor(corrupted_data, dtype=torch.float32),
                'target': torch.tensor(data, dtype=torch.float32),
                'meta': (data_min, data_max)
            }
        except Exception as e:
            raise RuntimeError(f"Ошибка при обработке файла {file_path}: {e}")

class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, output_size=(800, 800)):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )
        self.final_upsample = nn.Upsample(size=output_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.decoder(x)
        return self.final_upsample(x)

class CombinedModel(nn.Module):
    def __init__(self, pretrained=True, swin_model_name='swin_tiny_patch4_window7_224', output_size=(800, 800)):
        super().__init__()
        self.encoder = create_model(
            swin_model_name,
            pretrained=False,
            img_size=224
        )
        self.encoder.head = nn.Identity()

        if pretrained:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            weights_path = os.path.join(current_dir, "models", "model.safetensors")
            state_dict = load_file(weights_path)
            filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
            self.encoder.load_state_dict(filtered_dict, strict=False)

        self.decoder = UNetDecoder(in_channels=768, out_channels=1, output_size=output_size)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        enc_features = self.encoder.forward_features(x)

        # Исправленная обработка признаков
        if isinstance(enc_features, (list, tuple)):
            enc_features = enc_features[-1]

        if len(enc_features.shape) == 4:
            # Формат [B, C, H, W] - пропускаем преобразование
            pass
        else:
            # Формат [B, N, C] -> [B, C, H, W]
            B, N, C = enc_features.shape
            H = W = int(N**0.5)
            enc_features = enc_features.permute(0, 2, 1).view(B, C, H, W)

        return self.decoder(enc_features)

# class CombinedModel(nn.Module):
#     def __init__(self, pretrained=True, swin_model_name='swin_tiny_patch4_window7_224', output_size=(800, 800)):
#         super().__init__()
#         # Используем оригинальную tiny версию
#         self.encoder = create_model(
#             swin_model_name,
#             pretrained=False,
#             img_size=224  # Возвращаем оригинальный размер
#         )
#         self.encoder.head = nn.Identity()
#
#         if pretrained:
#             current_dir = os.path.dirname(os.path.abspath(__file__))
#             weights_path = os.path.join(current_dir, "models", "model.safetensors")
#             state_dict = load_file(weights_path)
#
#             # Фильтруем несовместимые ключи
#             filtered_dict = {k: v for k, v in state_dict.items()
#                              if not k.startswith('head')}
#
#             self.encoder.load_state_dict(filtered_dict, strict=False)
#
#         # Корректируем входные каналы для tiny версии
#         self.decoder = UNetDecoder(
#             in_channels=768,
#             out_channels=1,
#             output_size=output_size
#         )
#
#     def forward(self, x):
#         if x.shape[1] == 1:
#             x = x.repeat(1, 3, 1, 1)
#
#         # Возвращаем оригинальный размер 224x224
#         x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
#         enc_features = self.encoder.forward_features(x)
#
#         # Обработка признаков для tiny версии
#         if isinstance(enc_features, (list, tuple)):
#             enc_features = enc_features[-1]
#
#         B, N, C = enc_features.shape
#         H = W = int(N**0.5)
#         enc_features = enc_features.permute(0, 2, 1).view(B, C, H, W)
#
#         return self.decoder(enc_features)

# class CombinedModel(nn.Module):
#     def __init__(self, pretrained=True, swin_model_name='swin_base_patch4_window12_384', output_size=(800, 800)):
#         super().__init__()
#         # Используем модель для большего разрешения
#         self.encoder = create_model(swin_model_name, pretrained=False, img_size=384)
#         self.encoder.head = nn.Identity()
#
#         if pretrained:
#             current_dir = os.path.dirname(os.path.abspath(__file__))
#             weights_path = os.path.join(current_dir, "models", "model.safetensors")
#             state_dict = load_file(weights_path)
#             msg = self.encoder.load_state_dict(state_dict, strict=False)
#             print("Weights loading status:", msg)
#
#         self.decoder = UNetDecoder(
#             in_channels=1024 if 'base' in swin_model_name else 768,
#             out_channels=1,
#             output_size=output_size
#         )
#
#     def forward(self, x):
#         if x.shape[1] == 1:
#             x = x.repeat(1, 3, 1, 1)
#
#         # Увеличиваем размер входного изображения для Swin
#         x = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
#         enc_features = self.encoder.forward_features(x)
#
#         if isinstance(enc_features, (list, tuple)):
#             enc_features = enc_features[-1]
#
#         # Решейпинг признаков
#         B, N, C = enc_features.shape
#         H = W = int(N**0.5)
#         enc_features = enc_features.permute(0, 2, 1).view(B, C, H, W)
#
#         return self.decoder(enc_features)

def mixed_loss(output, target):
    ssim_value = torch.mean(torch.ssim(output, target, data_range=1.0))
    l1_loss = F.l1_loss(output, target)
    return 0.7*(1 - ssim_value) + 0.3*l1_loss

def visualize_results(corrupted, output, target, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Corrupted', 'Output', 'Target']

    for i, (data, title) in enumerate(zip([corrupted, output, target], titles)):
        axes[i].imshow(data.cpu().squeeze().numpy(), cmap='seismic', aspect='auto')
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.savefig(os.path.join(save_dir, f'epoch_{epoch+1}.png'))
    plt.close()

def train_model(swin_model_name='swin_base_patch4_window12_384'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    save_dir = os.path.join(current_dir, "results")

    dataset = SeismicDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # model = CombinedModel(pretrained=True, swin_model_name=swin_model_name)
    model = CombinedModel(pretrained=True)  # Без указания модели
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    best_loss = float('inf')
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in pbar:
                corrupted = batch['corrupted'].to(device).unsqueeze(1)
                target = batch['target'].to(device).unsqueeze(1)

                optimizer.zero_grad()
                output = model(corrupted)

                loss = mixed_loss(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

                # Визуализация первого батча
                if pbar.n == 0:
                    visualize_results(
                        corrupted[0],
                        output[0].detach(),
                        target[0],
                        epoch,
                        save_dir
                    )

        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)

        # Сохранение лучшей модели
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(current_dir, "best_model.pth"))

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

if __name__ == "__main__":
    train_model()