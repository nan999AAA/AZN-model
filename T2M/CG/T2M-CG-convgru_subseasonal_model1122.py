# T2M-CG-convgru_subseasonal_model1122
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

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
DROPOUT_RATE = 0.1

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
# 加载数据并标准化处理输入
# ===========================
ds = xr.open_dataset(DATA_PATH)
for var in input_vars + target_vars:
    if ds[var].isnull().any():
        print(f"[NaN Warning] 变量 {var} 中存在 NaN，将填充为 0")
        ds[var] = ds[var].fillna(0)

# 计算均值和标准差用于归一化
mean_std = {}
for var in input_vars:
    mean = ds[var].mean().values
    std = ds[var].std().values
    std = std if std != 0 else 1
    ds[var] = (ds[var] - mean) / std
    mean_std[var] = (mean, std)

# ===========================
# 数据集
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

dataset = SubseasonalDataset(ds, input_vars, target_vars)
train_size = int(0.8 * len(dataset))
train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ===========================
# ConvGRU 单元定义
# ===========================
class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 3 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        z, r, h_hat = torch.chunk(gates, 3, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        h_hat = torch.tanh(r * h_hat)
        h = (1 - z) * h_prev + z * h_hat
        return h

# ===========================
# ConvGRU 网络模型
# ===========================
class ConvGRUNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.encoder = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.rnn = ConvGRUCell(hidden_channels, hidden_channels, kernel_size=kernel_size)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.decoder = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)

    def forward(self, x):
        # 假设x: (B, C, H, W)
        x = self.encoder(x)
        h = torch.zeros_like(x)
        h = self.rnn(x, h)
        h = self.dropout(h)
        out = self.decoder(h)
        return out

model = ConvGRUNet(input_channels=len(input_vars),
                   hidden_channels=64,
                   output_channels=len(target_vars)).to(DEVICE)

# ===========================
# 损失函数与优化器
# ===========================
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===========================
# 训练与验证
# ===========================
train_loss_list, val_loss_list, rmse_list, acc_list = [], [], [], []

def compute_rmse(pred, truth):
    return torch.sqrt(torch.mean((pred - truth)**2)).item()

def compute_acc(pred, truth):
    numerator = torch.sum((pred - truth)**2)
    denominator = torch.sum((truth - torch.mean(truth))**2)
    return 1 - numerator / denominator if denominator != 0 else torch.tensor(0.0)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        if torch.isnan(loss):
            print("[Warning] Loss is NaN，跳过该 batch")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    train_loss_list.append(total_loss / len(train_loader))

    # 验证
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

# ===========================
# 保存模型
# ===========================
torch.save(model.state_dict(), './convgru_subseasonal_model1122.pth')