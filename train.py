import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# Для вычисления SSIM используем pytorch_msssim (его можно установить через pip)
from pytorch_msssim import ssim

# -------------------------------
# 1. ПОДГОТОВКА ДАННЫХ
# -------------------------------
class PairedSeismicDataset(Dataset):
    """
    Датасет для загрузки пар сейсмограмм:
    - X: сейсмограммы с прореженными трассами (папка "subsampled")
    - Y: полные, чистые сейсмограммы (папка "full")
    Предполагается, что каждая сейсмограмма сохранена в формате .npy и представляет собой 2D-массив.
    Если изображение больше, чем patch_size, то выбирается случайная обрезка нужного размера.
    """
    def __init__(self, input_dir="subsampled", target_dir="full", patch_size=64):
        super(PairedSeismicDataset, self).__init__()
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.patch_size = patch_size

        # Загружаем список файлов из обеих папок
        self.input_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
        self.target_files = sorted(glob.glob(os.path.join(target_dir, "*.npy")))

        if len(self.input_files) == 0 or len(self.target_files) == 0:
            raise ValueError("Не найдены файлы в одной из указанных папок!")
        if len(self.input_files) != len(self.target_files):
            raise ValueError("Количество файлов в папке с входными данными и целевыми не совпадает!")

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Загружаем входное изображение (с прореженными трассами) и целевое изображение (полное)
        input_img = np.load(self.input_files[idx]).astype(np.float32)
        target_img = np.load(self.target_files[idx]).astype(np.float32)
        H, W = input_img.shape

        # Если изображение больше, чем patch_size, выбираем случайную обрезку
        if H > self.patch_size and W > self.patch_size:
            i = np.random.randint(0, H - self.patch_size + 1)
            j = np.random.randint(0, W - self.patch_size + 1)
            input_img = input_img[i:i+self.patch_size, j:j+self.patch_size]
            target_img = target_img[i:i+self.patch_size, j:j+self.patch_size]
        else:
            # Если изображение меньше или равно patch_size, берём верхний левый угол
            input_img = input_img[:self.patch_size, :self.patch_size]
            target_img = target_img[:self.patch_size, :self.patch_size]

        # Преобразуем в тензоры и добавляем размерность канала
        input_tensor = torch.from_numpy(input_img).unsqueeze(0)
        target_tensor = torch.from_numpy(target_img).unsqueeze(0)
        return input_tensor, target_tensor

# -------------------------------
# 2. ОПРЕДЕЛЕНИЕ АРХИТЕКТУРЫ SIT
# -------------------------------
# 2.1 Head Block (HB)
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

# 2.2 Общий блок Transformer (для ET и DT)
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

# Функции для преобразования карты признаков в последовательность и обратно
def flatten_features(x):
    B, C, H, W = x.size()
    return x.view(B, C, H * W).permute(0, 2, 1)

def unflatten_features(x, H, W):
    B, N, C = x.size()
    return x.permute(0, 2, 1).view(B, C, H, W)

# 2.3 Encoder Transformer (ET)
class EncoderTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=8, ff_dim=256):
        super(EncoderTransformer, self).__init__()
        self.transformer = TransformerBlock(d_model=d_model, nhead=nhead, ff_dim=ff_dim)

    def forward(self, x):
        B, C, H, W = x.size()
        x_seq = flatten_features(x)
        x_seq = self.transformer(x_seq)
        x_out = unflatten_features(x_seq, H, W)
        return x_out

# 2.4 U-shaped Swin-Transformer (UST) блок
class USTBlock(nn.Module):
    """
U-shaped блок с downsampling, TransformerBlock и upsampling.
    """
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

# 2.5 Decoder Transformer (DT)
class DecoderTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=8, ff_dim=256):
        super(DecoderTransformer, self).__init__()
        self.transformer = TransformerBlock(d_model=d_model, nhead=nhead, ff_dim=ff_dim)

    def forward(self, x, skip):
        B, C, H, W = x.size()
        x = x + skip  # Добавляем skip-соединение
        x_seq = flatten_features(x)
        x_seq = self.transformer(x_seq)
        x_out = unflatten_features(x_seq, H, W)
        return x_out

# 2.6 Tail Block (TB): пространственное внимание + 1x1 свёртка
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
        x_weighted = x * attn_weights
        out = self.final_conv(x_weighted)
        return out

# 2.7 Собираем всю архитектуру SIT
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
        out = self.tail(x_dec)
        return out

# -------------------------------
# 3. ФУНКЦИЯ ПОТЕРЬ L1 + SSIM
# -------------------------------
def l1_ssim_loss(pred, target, alpha=1.0, beta=1.0):
    l1_loss = F.l1_loss(pred, target)
    ssim_val = ssim(pred, target, data_range=target.max() - target.min(), size_average=True)
    loss = alpha * l1_loss + beta * (1 - ssim_val)
    return loss

# -------------------------------
# 4. ТРЕНИРОВКА И ВАЛИДАЦИЯ
# -------------------------------
def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-3,
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_losses = []
    val_losses = []
    train_ssim_vals = []
    val_ssim_vals = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_ssim = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, targets in train_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = l1_ssim_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            batch_ssim = ssim(outputs, targets, data_range=targets.max() - targets.min(), size_average=True).item()
            running_ssim += batch_ssim * inputs.size(0)
            train_bar.set_postfix(loss=loss.item(), ssim=batch_ssim)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_ssim = running_ssim / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_ssim_vals.append(epoch_ssim)

        model.eval()
        val_running_loss = 0.0
        val_running_ssim = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = l1_ssim_loss(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)
                batch_ssim = ssim(outputs, targets, data_range=targets.max() - targets.min(), size_average=True).item()
                val_running_ssim += batch_ssim * inputs.size(0)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_ssim = val_running_ssim / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        val_ssim_vals.append(val_epoch_ssim)
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f}, Train SSIM: {epoch_ssim:.4f} | Val Loss: {val_epoch_loss:.4f}, Val SSIM: {val_epoch_ssim:.4f}")
        scheduler.step()

        if (epoch+1) % 10 == 0:
            sample_input, sample_target = next(iter(val_loader))
            sample_input = sample_input.to(device)
            sample_output = model(sample_input)
            sample_input_np = sample_input[0,0].detach().cpu().numpy()
            sample_output_np = sample_output[0,0].detach().cpu().numpy()
            sample_target_np = sample_target[0,0].detach().cpu().numpy()

            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.title("Input (subsampled)")
            plt.imshow(sample_input_np, cmap='gray')
            plt.subplot(1,3,2)
            plt.title("Output (восстановлено)")
            plt.imshow(sample_output_np, cmap='gray')
            plt.subplot(1,3,3)
            plt.title("Target (full)")
            plt.imshow(sample_target_np, cmap='gray')
            plt.show()

    epochs = np.arange(1, num_epochs+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, train_ssim_vals, label='Train SSIM')
    plt.plot(epochs, val_ssim_vals, label='Val SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.show()

    return model, train_losses, val_losses, train_ssim_vals, val_ssim_vals

# -------------------------------
# 5. ТЕСТИРОВАНИЕ НА ОТЛОЖЕННЫХ ДАННЫХ
# -------------------------------
def test_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_ssim = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = l1_ssim_loss(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            batch_ssim = ssim(outputs, targets, data_range=targets.max() - targets.min(), size_average=True).item()
            total_ssim += batch_ssim * inputs.size(0)

    avg_loss = total_loss / len(test_loader.dataset)
    avg_ssim = total_ssim / len(test_loader.dataset)
    print(f"Test Loss: {avg_loss:.4f}, Test SSIM: {avg_ssim:.4f}")

    sample_inputs, sample_targets = next(iter(test_loader))
    sample_inputs = sample_inputs.to(device)
    sample_outputs = model(sample_inputs)
    for i in range(min(3, sample_inputs.size(0))):
        inp = sample_inputs[i,0].detach().cpu().numpy()
        out = sample_outputs[i,0].detach().cpu().numpy()
        tar = sample_targets[i,0].detach().cpu().numpy()
        plt.figure(figsize=(10,3))
        plt.subplot(1,3,1)
        plt.title("Input")
        plt.imshow(inp, cmap='gray')
        plt.subplot(1,3,2)
        plt.title("Output")
        plt.imshow(out, cmap='gray')
        plt.subplot(1,3,3)
        plt.title("Target")
        plt.imshow(tar, cmap='gray')
        plt.show()

# -------------------------------
# 6. ОСНОВНАЯ ФУНКЦИЯ
# -------------------------------
def main():
    # Гиперпараметры
    # input_dir = "subsampled"  # Папка с 500 прореженными сейсмограммами
    # target_dir = "full"       # Папка с 500 полными сейсмограммами
    # Path to the folder containing seismic data
    # data_folder = "test_data/subsampled"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, "data")
    input_dir = os.path.join(current_dir, "subsampled")
    patch_size = 64
    batch_size = 8
    num_epochs = 10
    lr = 1e-3

    # Создаем датасет на основе пар файлов
    dataset = PairedSeismicDataset(input_dir=input_dir, target_dir=target_dir, patch_size=patch_size)

    # Разбиваем на train/val/test (например, 80%/10%/10%)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Инициализируем модель
    model = SITModel()
    print("Модель SIT:")
    print(model)

    # Обучение
    model, train_losses, val_losses, train_ssim, val_ssim = train_model(model, train_loader, val_loader,
                                                                        num_epochs=num_epochs, lr=lr)

    # Сохраняем модель
    torch.save(model.state_dict(), "SIT_model.pth")


    # Тестирование
    test_model(model, test_loader)

if __name__ == '__main__':
    main()
    
