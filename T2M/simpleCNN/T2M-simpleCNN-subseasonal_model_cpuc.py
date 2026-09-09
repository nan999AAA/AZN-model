# 1 T2M-simpleCNN-subseasonal_model_cpuccs
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

# 当前代码文件所在目录：T2M/simpleCNN/
SCRIPT_DIR = Path(__file__).resolve().parent

# 上一级目录：T2M/
PROJECT_DIR = SCRIPT_DIR.parent

# ===========================
# 配置
# ===========================
# 数据文件：T2M/D.nc
DATA_PATH = PROJECT_DIR / "D.nc"
DEVICE = torch.device("cpu")
BATCH_SIZE = 2
EPOCHS = 100
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
# 加载并清洗数据
# ===========================
ds = xr.open_dataset(DATA_PATH)

# 检查并填充 NaN（防止训练异常）
# 原代码（NaN填0）
for var in input_vars + target_vars:
    if ds[var].isnull().any():
        print(f"[NaN Warning] 变量 {var} 中存在 NaN，将填充为 0")
        ds[var] = ds[var].fillna(0)


# 可选：检查并删除所有含 NaN 的时间步
# valid_mask = ~np.isnan(ds[input_vars + target_vars].to_array().values).any(axis=(0, 2, 3))
# ds = ds.isel(time=valid_mask)

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
# 模型定义（加初始化）
# ===========================
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.model:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)

model = SimpleCNN(len(input_vars), len(target_vars)).to(DEVICE)
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

        # 防止NaN传递
        if torch.isnan(loss):
            print("[Warning] Loss is NaN. 跳过该 batch")
            continue

        loss.backward()

        # 加入梯度裁剪
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

# ===========================
# 时间序列级别指标记录
# ===========================
time_metrics = {
    var: {'loss': [], 'rmse': [], 'acc': []}
    for var in target_vars
}
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

        # 记录时间索引（假设 Dataset 中 time 是 DataArray）
        time_indices.append(ds.time.values[train_size + idx])

# ===========================
# 保存模型
# ===========================
torch.save(model.state_dict(), './subseasonal_model_cpucc.pth')
