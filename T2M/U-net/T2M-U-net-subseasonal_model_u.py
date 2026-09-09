#T2M-U-net-subseasonal_model_u
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
import torch.nn.functional as F
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
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 1e-3
CHECKPOINT_PATH = './subseasonal_model_u.pth'
EPOCH_RECORD_FILE = './checkpoint_epoch.txt'  # 用于保存断点epoch信息

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

dataset = SubseasonalDataset(ds, input_vars, target_vars)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ===========================
# UNet带残差块和BatchNorm定义及辅助函数
# ===========================
def center_crop(tensor, target_size):
    _, _, h, w = tensor.size()
    th, tw = target_size
    x1 = (h - th) // 2
    y1 = (w - tw) // 2
    return tensor[:, :, x1:x1+th, y1:y1+tw]

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        if in_c != out_c:
            self.residual = nn.Conv2d(in_c, out_c, 1)
        else:
            self.residual = nn.Identity()
    def forward(self, x):
        res = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.relu(out)

class UNet_Res(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.enc1 = ResidualBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ResidualBlock(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ResidualBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ResidualBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ResidualBlock(128, 64)

        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)         # (B,64,H,W)
        p1 = self.pool1(e1)      # (B,64,H/2,W/2)

        e2 = self.enc2(p1)       # (B,128,H/2,W/2)
        p2 = self.pool2(e2)      # (B,128,H/4,W/4)

        e3 = self.enc3(p2)       # (B,256,H/4,W/4)
        p3 = self.pool3(e3)      # (B,256,H/8,W/8)

        b = self.bottleneck(p3)  # (B,512,H/8,W/8)

        u3 = self.up3(b)         # (B,256,H/4,W/4)
        if u3.size()[2:] != e3.size()[2:]:
            e3 = center_crop(e3, u3.size()[2:])
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)        # (B,128,H/2,W/2)
        if u2.size()[2:] != e2.size()[2:]:
            e2 = center_crop(e2, u2.size()[2:])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)        # (B,64,H,W)
        if u1.size()[2:] != e1.size()[2:]:
            e1 = center_crop(e1, u1.size()[2:])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.output_layer(d1)

        if out.shape[2] != 121 or out.shape[3] != 240:
            out = F.interpolate(out, size=(121,240), mode='bilinear', align_corners=False)
        return out

model = UNet_Res(len(input_vars), len(target_vars)).to(DEVICE)

# ===========================
# 损失函数和优化器
# ===========================
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===========================
# 断点续训相关函数
# ===========================
def save_checkpoint(epoch):
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    with open(EPOCH_RECORD_FILE, 'w') as f:
        f.write(str(epoch))

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH) and os.path.exists(EPOCH_RECORD_FILE):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        with open(EPOCH_RECORD_FILE, 'r') as f:
            start_epoch = int(f.read())
        print(f"从断点 epoch {start_epoch} 继续训练")
        return start_epoch
    else:
        print("未找到断点，重新开始训练")
        return 0

# ===========================
# 训练与验证函数
# ===========================
train_loss_list, val_loss_list, rmse_list, acc_list = [], [], [], []

def compute_rmse(pred, truth):
    return torch.sqrt(torch.mean((pred - truth)**2)).item()

def compute_acc(pred, truth):
    numerator = torch.sum((pred - truth)**2)
    denominator = torch.sum((truth - torch.mean(truth))**2)
    return 1 - numerator / denominator if denominator != 0 else torch.tensor(0.0)

start_epoch = load_checkpoint()

for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)

        if torch.isnan(loss):
            print("[Warning] Loss is NaN. 跳过该 batch")
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
                print("[Val Warning] Loss is NaN. 跳过该验证样本")
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

    save_checkpoint(epoch+1)

# ===========================
# 时间序列级别指标记录
# ===========================
time_metrics = {var: {'loss': [], 'rmse': [], 'acc': []} for var in target_vars}
time_indices = []

model.eval()
with torch.no_grad():
    for idx in range(len(val_ds)):
        x, y = val_ds[idx]
        x = x.unsqueeze(0).to(DEVICE)
        y = y.unsqueeze(0).to(DEVICE)
        pred = model(x)

        for var_idx, var in enumerate(target_vars):
            pred_var = pred[:, var_idx, :, :]
            y_var = y[:, var_idx, :, :]
            loss = loss_fn(pred_var, y_var)
            rmse = compute_rmse(pred_var, y_var)
            acc = compute_acc(pred_var, y_var)
            time_metrics[var]['loss'].append(loss.item())
            time_metrics[var]['rmse'].append(rmse)
            time_metrics[var]['acc'].append(acc.item())

        time_indices.append(ds.time.values[train_size + idx])

# ===========================
# 最终保存模型
# ===========================
torch.save(model.state_dict(), CHECKPOINT_PATH)

