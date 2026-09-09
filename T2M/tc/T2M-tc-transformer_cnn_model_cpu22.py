# T2M-tc-transformer_cnn_model_cpu22
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
EPOCHS = 99
LEARNING_RATE = 1e-3

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
# 加载数据
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

dataset = SubseasonalDataset(ds, input_vars, target_vars)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ===========================
# Transformer + CNN 模型定义
# ===========================
class TransformerCNN(nn.Module):
    def __init__(self, in_channels, out_channels, height, width, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.height = height
        self.width = width
        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x shape: (B, C, H, W)
        x = self.patch_embed(x)  # (B, d_model, H, W)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # (B, H*W, d_model)
        x = self.transformer(x)  # (B, H*W, d_model)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)  # (B, d_model, H, W)
        out = self.decoder(x)  # (B, out_channels, H, W)
        return out

# ===========================
# 模型初始化
# ===========================
sample_x, _ = dataset[0]
_, H, W = sample_x.shape
model = TransformerCNN(len(input_vars), len(target_vars), H, W).to(DEVICE)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===========================
# 训练与验证循环
# ===========================
train_loss_list, val_loss_list, rmse_list, acc_list = [], [], [], []

def compute_rmse(pred, truth):
    return torch.sqrt(torch.mean((pred - truth) ** 2)).item()

def compute_acc(pred, truth):
    numerator = torch.sum((pred - truth) ** 2)
    denominator = torch.sum((truth - torch.mean(truth)) ** 2)
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
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

# ===========================
# 保存模型
# ===========================
torch.save(model.state_dict(), './transformer_cnn_model_cpu22.pth')
