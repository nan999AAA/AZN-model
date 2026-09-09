# T2M&MSLP MONTH
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# === 参数 ===
timesteps = 15
variables = ["u10", "v10", "sst", "t2m", "msl"]
target_indices = [variables.index("t2m"), variables.index("msl")]
data_path = "merged_1.5deg_no_norm.nc"
batch_size = 4
epochs = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备: {device}")

# === 加载原始数据 ===
ds = xr.open_dataset(data_path)
H, W = ds[variables[0]].shape[1:]
data_stack = np.stack([ds[var].values for var in variables], axis=1)  # (T, 5, H, W)

# === 添加时间编码 ===
month = (ds.time.dt.month.values - 1).astype(np.float32)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
month_sin_2d = np.tile(month_sin[:, None, None, None], (1, 1, H, W))
month_cos_2d = np.tile(month_cos[:, None, None, None], (1, 1, H, W))
data_stack = np.concatenate([data_stack, month_sin_2d, month_cos_2d], axis=1)  # (T, 7, H, W)

# === 标准化（前5个变量）===
mean = data_stack[:, :5].mean(axis=(0, 2, 3), keepdims=True)
std = data_stack[:, :5].std(axis=(0, 2, 3), keepdims=True)
std[std == 0] = 1.0
data_stack[:, :5] = (data_stack[:, :5] - mean) / std

# === Dataset ===
class ClimateDataset(Dataset):
    def __init__(self, data, timesteps, start_idx=0, end_idx=None):
        self.data = data
        self.timesteps = timesteps
        self.start_idx = start_idx
        self.end_idx = end_idx or (len(data) - timesteps - 2)  # 预测2个月

    def __len__(self):
        return self.end_idx - self.start_idx + 1

    def __getitem__(self, idx):
        idx += self.start_idx
        x = self.data[idx:idx + self.timesteps]  # (T, 7, H, W)
        y = self.data[idx + self.timesteps: idx + self.timesteps + 2, target_indices]  # (2, 2, H, W)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# === 数据划分 ===
total_samples = data_stack.shape[0] - timesteps - 2
train_size = int(total_samples * 0.8)
train_dataset = ClimateDataset(data_stack, timesteps, 0, train_size)
test_dataset = ClimateDataset(data_stack, timesteps, train_size, total_samples)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# === 模型 ===
class ChEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Linear(32 * 4 * 4, embed_dim)

    def forward(self, x):
        x = self.spatial_encoder(x)
        x = x.flatten(1)
        return self.fc(x)

class TemporalConvNet(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=dilation, dilation=dilation),
                nn.ReLU(), nn.Dropout(dropout)
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):  # (B, C, T)
        return self.network(x)

class ClimateTCNModel(nn.Module):
    def __init__(self, input_channels, height, width, timesteps, embed_dim=64):
        super().__init__()
        self.encoder = ChEncoder(input_channels, embed_dim)
        self.tcn = TemporalConvNet(embed_dim, [128, 128])
        self.fc = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 2 * 2 * height * width)  # 2月 × 2变量
        )
        self.height = height
        self.width = width

    def forward(self, x):  # (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.encoder(x)
        x = x.view(B, T, -1).transpose(1, 2)
        x = self.tcn(x)
        x = x[:, :, -1]
        out = self.fc(x)
        return out.view(B, 2, 2, H, W)  # (B, time, var, H, W)

# === 损失函数 ===
def weighted_mse_loss(pred, target, weight_factor=5.0):
    weight = torch.abs(target)
    weight = 1 + weight_factor * (weight / weight.max())
    return ((pred - target) ** 2 * weight).mean()

# === 初始化模型与优化器 ===
H, W = data_stack.shape[2:]
model = ClimateTCNModel(input_channels=7, height=H, width=W, timesteps=timesteps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === 恢复断点（如果存在）===
checkpoint_path = "checkpoint.pth"
start_epoch = 0
best_loss = float("inf")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["best_loss"]
    print(f"🔄 恢复断点：从 Epoch {start_epoch} 开始训练（Best Loss: {best_loss:.4f}）")

# === 日志初始化 ===
log_lines = []

# === 开始训练 ===
print("🚀 开始训练")
for epoch in range(start_epoch, epochs):
    model.train()
    total_loss = 0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = weighted_mse_loss(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    epoch_loss = total_loss / len(train_loader)
    log_line = f"✅ Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f}"
    print(log_line)
    log_lines.append(log_line)

    # 保存最佳模型
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "climate_tcn_with_timeencoding_150_2month.pth")
        print(f"🌟 当前为最优模型，保存为 climate_tcn_with_timeencoding_150_2month.pth（Loss: {best_loss:.4f}）")

    # 保存断点
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss
    }, checkpoint_path)

# === 写入日志 ===
with open("train_log.txt", "w") as f:
    for line in log_lines:
        f.write(line + "\n")

print("📄 训练日志已保存为 train_log.txt")
