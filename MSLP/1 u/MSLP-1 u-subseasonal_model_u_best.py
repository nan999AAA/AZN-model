# MSLP-1 u-subseasonal_model_u_best
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 禁用 GPU

import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_DIR = SCRIPT_DIR.parent

# ===========================
# 配置
# ===========================

DATA_PATH = PROJECT_DIR / "mslp_2D.nc"
DEVICE = torch.device("cpu")
BATCH_SIZE = 4
EPOCHS = 3000
LEARNING_RATE = 1e-3
CHECKPOINT_PATH = './subseasonal_model_u.pth'
BEST_MODEL_PATH = './subseasonal_model_u_best.pth'
EPOCH_RECORD_FILE = './checkpoint_epoch.txt'  # 用于保存断点epoch信息
NORM_PARAM_FILE = './norm_params.npz'  # 保存归一化参数

# ===========================
# 输入输出变量
# ===========================
input_vars = [
    # mslp 历史 20 周
    *[f"mslp_hist_{i}" for i in range(20)],

    # z850 历史 10 周
    *[f"z850_hist_{i}" for i in range(10)],

    # z500 - z850 历史 10 周
    *[f"z500_z850_hist_{i}" for i in range(10)],

    # q700 历史 10 周
    *[f"q700_hist_{i}" for i in range(10)],

    # divergence900 历史 10 周
    *[f"divergence900_hist_{i}" for i in range(10)],

    # pv900 历史 10 周
    *[f"pv900_hist_{i}" for i in range(10)],

    # 额外因子
    "pred_msl_month",
    "pred_t2m_month",
    "elevation"
]

target_vars = [
    "mslp_mon", "mslp_wed", "mslp_fri", "mslp_sun",
    "mslp_tue_next", "mslp_thu_next", "mslp_sat_next"
]

# ===========================
# 加载并清洗数据
# ===========================
ds = xr.open_dataset(DATA_PATH)
for var in input_vars + target_vars:
    if ds[var].isnull().any():
        print(f"[NaN Warning] 变量 {var} 中存在 NaN，将填充为 0")
        ds[var] = ds[var].fillna(0)

# ===========================
# 数据集定义（带输入输出标准化）
# ===========================
class SubseasonalDataset(Dataset):
    def __init__(self, ds, input_vars, target_vars, normalize=True, norm_params=None):
        x_data = ds[input_vars].to_array().transpose('time', 'variable', 'latitude', 'longitude')
        y_data = ds[target_vars].to_array().transpose('time', 'variable', 'latitude', 'longitude')

        if normalize:
            if norm_params is None:
                # 输入标准化参数
                self.x_mean = x_data.mean(dim=('time', 'latitude', 'longitude'))
                self.x_std = x_data.std(dim=('time', 'latitude', 'longitude'))
                self.x_std = xr.where(self.x_std == 0, 1.0, self.x_std)
                # 输出标准化参数
                self.y_mean = y_data.mean(dim=('time', 'latitude', 'longitude'))
                self.y_std = y_data.std(dim=('time', 'latitude', 'longitude'))
                self.y_std = xr.where(self.y_std == 0, 1.0, self.y_std)
            else:
                self.x_mean = xr.DataArray(norm_params['x_mean'], dims=('variable',))
                self.x_std = xr.DataArray(norm_params['x_std'], dims=('variable',))
                self.y_mean = xr.DataArray(norm_params['y_mean'], dims=('variable',))
                self.y_std = xr.DataArray(norm_params['y_std'], dims=('variable',))

            x_data = (x_data - self.x_mean) / self.x_std
            y_data = (y_data - self.y_mean) / self.y_std
        else:
            self.x_mean = None
            self.x_std = None
            self.y_mean = None
            self.y_std = None

        self.x = x_data
        self.y = y_data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx].values, dtype=torch.float32),
            torch.tensor(self.y[idx].values, dtype=torch.float32)
        )

# ===========================
# 创建训练集（加载或保存标准化参数）
# ===========================
if os.path.exists(NORM_PARAM_FILE):
    params = np.load(NORM_PARAM_FILE)
    dataset = SubseasonalDataset(
        ds, input_vars, target_vars,
        normalize=True,
        norm_params={
            'x_mean': params['x_mean'],
            'x_std': params['x_std'],
            'y_mean': params['y_mean'],
            'y_std': params['y_std']
        }
    )
else:
    dataset = SubseasonalDataset(ds, input_vars, target_vars, normalize=True)
    np.savez(
        NORM_PARAM_FILE,
        x_mean=dataset.x_mean.values,
        x_std=dataset.x_std.values,
        y_mean=dataset.y_mean.values,
        y_std=dataset.y_std.values
    )
    print(f"[Info] 标准化参数已保存到 {NORM_PARAM_FILE}")

train_ds = dataset
val_ds = dataset
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ===========================
# UNet带残差块和BatchNorm
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
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)

        u3 = self.up3(b)
        if u3.size()[2:] != e3.size()[2:]:
            e3 = center_crop(e3, u3.size()[2:])
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        if u2.size()[2:] != e2.size()[2:]:
            e2 = center_crop(e2, u2.size()[2:])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.size()[2:] != e1.size()[2:]:
            e1 = center_crop(e1, u1.size()[2:])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.output_layer(d1)
        if out.shape[2] != 121 or out.shape[3] != 240:
            out = F.interpolate(out, size=(121,240), mode='bilinear', align_corners=False)
        return out

model = UNet_Res(len(input_vars), len(target_vars)).to(DEVICE)

# ===========================
# 损失 & 优化器
# ===========================
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===========================
# 断点续训
# ===========================
def save_checkpoint(epoch, model, optimizer):
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    with open(EPOCH_RECORD_FILE, 'w') as f:
        f.write(str(epoch))

def load_checkpoint(model, optimizer):
    if os.path.exists(CHECKPOINT_PATH) and os.path.exists(EPOCH_RECORD_FILE):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch']
        print(f"从断点 epoch {start_epoch} 继续训练")
        return start_epoch
    else:
        print("未找到断点，重新开始训练")
        return 0

# ===========================
# 指标
# ===========================
def compute_rmse(pred, truth):
    return torch.sqrt(torch.mean((pred - truth)**2)).item()

def compute_acc(pred, truth):
    numerator = torch.sum((pred - truth)**2)
    denominator = torch.sum((truth - torch.mean(truth))**2)
    return 1 - numerator / denominator if denominator != 0 else torch.tensor(0.0)

# ===========================
# 训练
# ===========================
best_acc = -float('inf')
start_epoch = load_checkpoint(model, optimizer)
train_loss_list, val_loss_list, rmse_list, acc_list = [], [], [], []

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

    save_checkpoint(epoch+1, model, optimizer)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"[Info] 新最佳模型保存于 epoch {epoch+1}，ACC={acc:.4f}")

