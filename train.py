import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from safetensors.torch import load_file
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


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
            data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-6)

            mask = np.ones_like(data)
            num_missing_traces = np.random.randint(5, 15)
            missing_indices = np.random.choice(data.shape[0], num_missing_traces, replace=False)
            mask[missing_indices, :] = 0
            corrupted_data = data * mask

            return torch.tensor(corrupted_data, dtype=torch.float32), torch.tensor(data, dtype=torch.float32)
        except Exception as e:
            raise RuntimeError(f"Ошибка при обработке файла {file_path}: {e}")


class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, output_size=(800, 800)):
        super(UNetDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512,kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        self.output_size = output_size  # Добавляем размер выхода

    def forward(self, x):
        x = self.decoder(x)
        # Масштабируем выход к исходному разрешению
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        return x


class CombinedModel(nn.Module):
    def __init__(self, pretrained=True, swin_model_name='swin_tiny_patch4_window7_224', output_size=(800, 800)):
        super(CombinedModel, self).__init__()
        # Создаем Swin Transformer
        self.encoder = create_model(swin_model_name, pretrained=False)
        self.encoder.head = nn.Identity()

        # Загрузка предобученных весов
        if pretrained:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            weights_path = os.path.join(current_dir, "models", "model.safetensors")
            state_dict = load_file(weights_path)
            self.encoder.load_state_dict(state_dict, strict=False)

        # U-Net декодер с указанием выходного разрешения
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
        else:
            raise ValueError(f"Unexpected output shape from encoder: {enc_features.shape}")

        output = self.decoder(enc_features)
        return output

def ssim_l1_loss(output, target):
    ssim_loss = 1 - F.cosine_similarity(output, target, dim=-1).mean()
    l1_loss = F.l1_loss(output, target)
    return 0.5 * (ssim_loss + l1_loss)


def train_model(swin_model_name='swin_tiny_patch4_window7_224'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    dataset = SeismicDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    model = CombinedModel(pretrained=True, swin_model_name=swin_model_name)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Снизьте learning rate


    num_epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for corrupted, target in dataloader:
                corrupted = corrupted.to(device).unsqueeze(1)
                target = target.to(device).unsqueeze(1)
                optimizer.zero_grad()

                output = model(corrupted)

                # Масштабируем target до размера output
                resized_target = F.interpolate(target, size=output.shape[2:], mode='bilinear', align_corners=False)

                # loss = ssim_l1_loss(output, resized_target)
                # loss = F.mse_loss(output, resized_target)
                ssim_loss = 1 - F.cosine_similarity(output, resized_target, dim=-1).mean()
                l2_loss = F.mse_loss(output, resized_target)
                loss = 0.8 * l2_loss + 0.2 * ssim_loss


                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {epoch_loss / len(dataloader):.4f}")

    model_save_path = os.path.join(current_dir, f"combined_model_{swin_model_name}_{num_epochs}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved as {model_save_path}")


if __name__ == "__main__":
    train_model(swin_model_name='swin_tiny_patch4_window7_224')