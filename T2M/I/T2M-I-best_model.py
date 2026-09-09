# T2M-I-best_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 禁用 GPU

import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# 上一级目录：T2M/
PROJECT_DIR = SCRIPT_DIR.parent

# ===========================
# 配置
# ===========================
# 数据文件：T2M/D.nc
DATA_PATH = PROJECT_DIR / "D.nc"
DEVICE = torch.device("cpu")
BATCH_SIZE = 1
EPOCHS = 150
LEARNING_RATE = 1e-3
CHECKPOINT_PATH = "./subseasonal_checkpoint.pth"
BEST_MODEL_PATH = "./best_model.pth"

# ===========================
# 输入输出变量
# ===========================
input_vars = [f'tas_hist_{i}' for i in range(20)] + \
             [f'gh200_hist_{i}' for i in range(10)] + \
             [f'gh200_300_hist_{i}' for i in range(10)] + \
             [f'gh200_500_hist_{i}' for i in range(10)] + \
             ['pred_t2m_month', 'elevation']
target_vars = ['tas_mon', 'tas_wed', 'tas_fri', 'tas_sun', 'tas_tue_next', 'tas_thu_next', 'tas_sat_next']

# ===========================
# 加载并清洗数据
# ===========================
ds = xr.open_dataset(DATA_PATH)
for var in input_vars + target_vars:
    if ds[var].isnull().any():
        print(f"[NaN Warning] 变量 {var} 中存在 NaN，将填充为 0")
        ds[var] = ds[var].fillna(0)

# ===========================
# 数据集定义
# ===========================
class SubseasonalDataset(Dataset):
    def __init__(self, ds, input_vars, target_vars):
        self.x = ds[input_vars].to_array().transpose('time', 'variable', 'latitude', 'longitude')
        self.y = ds[target_vars].to_array().transpose('time', 'variable', 'latitude', 'longitude')

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx].values, dtype=torch.float32),
            torch.tensor(self.y[idx].values, dtype=torch.float32)
        )

# ===========================
# 模型模块定义
# ===========================
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.se = SEBlock(out_channels)
        self.spatial = SpatialAttention()
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.se(out)
        out = self.spatial(out)
        out += identity
        return self.relu(out)

class ImprovedNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualSEBlock(in_channels, 64),
            ResidualSEBlock(64, 64),
            ResidualSEBlock(64, 32),
            ResidualSEBlock(32, 32)
        )
        self.decoder = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.layers(x)
        return self.decoder(x)

model = ImprovedNet(len(input_vars), len(target_vars)).to(DEVICE)

# ===========================
# 损失函数（MSE + SSIM）
# ===========================
def ssim_loss(pred, target, C1=0.01**2, C2=0.03**2):
    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x = pred.var()
    sigma_y = target.var()
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return 1 - numerator / denominator

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.alpha * self.mse(pred, target) + (1 - self.alpha) * ssim_loss(pred, target)

loss_fn = CombinedLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===========================
# 数据加载器
# ===========================
dataset = SubseasonalDataset(ds, input_vars, target_vars)
train_size = int(0.99 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ===========================
# 训练与验证（断点续训 + 最佳模型保存）
# ===========================
start_epoch = 0
train_loss_list, val_loss_list, rmse_list, acc_list = [], [], [], []
best_val_loss = float('inf')

# 加载断点
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_loss_list = checkpoint['train_loss_list']
    val_loss_list = checkpoint['val_loss_list']
    rmse_list = checkpoint['rmse_list']
    acc_list = checkpoint['acc_list']
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"[Resuming] 从第 {start_epoch} 轮开始续训，当前最小验证损失为 {best_val_loss:.4f}")

def compute_rmse(pred, truth):
    return torch.sqrt(torch.mean((pred - truth)**2)).item()

def compute_acc(pred, truth):
    numerator = torch.sum((pred - truth)**2)
    denominator = torch.sum((truth - torch.mean(truth))**2)
    return 1 - numerator / denominator if denominator != 0 else torch.tensor(0.0)

for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    train_loss_list.append(total_loss / len(train_loader))

    model.eval()
    val_loss, rmse, acc = 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            batch_loss = loss_fn(pred, y)
            if torch.isnan(batch_loss):
                continue
            val_loss += batch_loss.item()
            rmse += compute_rmse(pred, y)
            acc += compute_acc(pred, y).item()
    val_loss /= len(val_loader)
    rmse /= len(val_loader)
    acc /= len(val_loader)
    val_loss_list.append(val_loss)
    rmse_list.append(rmse)
    acc_list.append(acc)
    print(f"Epoch {epoch+1}: Train Loss={train_loss_list[-1]:.4f} | Val Loss={val_loss:.4f} | RMSE={rmse:.4f} | ACC={acc:.4f}")

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"[Saved] Best model updated at epoch {epoch+1}, val_loss={val_loss:.4f}")

    # 保存断点
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_list': train_loss_list,
        'val_loss_list': val_loss_list,
        'rmse_list': rmse_list,
        'acc_list': acc_list,
        'best_val_loss': best_val_loss
    }, CHECKPOINT_PATH)

