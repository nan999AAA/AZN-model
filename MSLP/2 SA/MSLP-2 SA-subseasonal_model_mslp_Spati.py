# MSLP-2 SA-subseasonal_model_mslp_SpatialAttentionResNet9992-2
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

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_DIR = SCRIPT_DIR.parent

# ===========================
# 配置
# ===========================

DATA_PATH = PROJECT_DIR / "mslp_2D.nc"
DEVICE = torch.device("cpu")
BATCH_SIZE = 1
EPOCHS = 150
LEARNING_RATE = 1e-3
SAVE_DIR = "./SpatialAttentionResNet-2"
os.makedirs(SAVE_DIR, exist_ok=True)

CKPT_LAST = os.path.join(SAVE_DIR, "last_checkpoint-2.pth")
CKPT_BEST = os.path.join(SAVE_DIR, "subseasonal_model_mslp_SpatialAttentionResNet9992-2.pth")

# ===========================
# 输入输出变量
# ===========================
input_vars = [
    *[f"mslp_hist_{i}" for i in range(20)],
    *[f"z850_hist_{i}" for i in range(10)],
    *[f"z500_z850_hist_{i}" for i in range(10)],
    *[f"q700_hist_{i}" for i in range(10)],
    *[f"divergence900_hist_{i}" for i in range(10)],
    *[f"pv900_hist_{i}" for i in range(10)],
    "pred_msl_month",
    "pred_t2m_month",
    "elevation"
]

target_vars = ['mslp_mon', 'mslp_wed', 'mslp_fri', 'mslp_sun',
               'mslp_tue_next', 'mslp_thu_next', 'mslp_sat_next']

# ===========================
# 加载数据
# ===========================
ds = xr.open_dataset(DATA_PATH)
for var in input_vars + target_vars:
    if var not in ds:
        raise KeyError(f"变量 {var} 不存在于数据集中，请检查")
    if ds[var].isnull().any():
        print(f"[NaN Warning] 变量 {var} 中存在 NaN，将填充为 0")
        ds[var] = ds[var].fillna(0)

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
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===========================
# 模型
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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.attn = SpatialAttention()
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.attn(out)
        out += identity
        return self.relu(out)

class SpatialAttentionResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(in_channels, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 32),
        )
        self.decoder = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = SpatialAttentionResNet(len(input_vars), len(target_vars)).to(DEVICE)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===========================
# 评价指标函数
# ===========================
def compute_rmse(pred, truth):
    return torch.sqrt(torch.mean((pred - truth)**2)).item()

def compute_acc(pred, truth):
    pred_flat = pred.view(-1)
    truth_flat = truth.view(-1)
    pred_anom = pred_flat - pred_flat.mean()
    truth_anom = truth_flat - truth_flat.mean()
    denom = torch.sqrt(torch.sum(pred_anom**2) * torch.sum(truth_anom**2))
    if denom == 0:
        return 0.0
    return (torch.sum(pred_anom * truth_anom) / denom).item()

def compute_nac(pred, truth):
    """Normalized anomaly correlation"""
    rmse = compute_rmse(pred, truth)
    truth_std = torch.std(truth)
    if truth_std == 0:
        return 0.0
    return 1 - (rmse / truth_std).item()

# ===========================
# 训练（batch 级别断点保存）
# ===========================
start_epoch = 0
best_acc = -float("inf")
train_loss_list, val_loss_list = [], []
val_rmse_list, val_acc_list, val_nac_list = [], [], []

# batch级别记录
batch_metrics = {
    "rmse": [],
    "acc": [],
    "nac": []
}

# 加载断点
if os.path.exists(CKPT_LAST):
    print(f"[Resume] 从 {CKPT_LAST} 恢复训练...")
    ckpt = torch.load(CKPT_LAST, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"] + 1
    best_acc = ckpt["best_acc"]
    train_loss_list = ckpt["train_loss_list"]
    val_loss_list = ckpt["val_loss_list"]
    val_rmse_list = ckpt["val_rmse_list"]
    val_acc_list = ckpt["val_acc_list"]
    val_nac_list = ckpt.get("val_nac_list", [])
    batch_metrics = ckpt.get("batch_metrics", {"rmse": [], "acc": [], "nac": []})

for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0

    for i, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Train")):
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

        # batch级别指标
        batch_rmse = compute_rmse(pred, y)
        batch_acc = compute_acc(pred, y)
        batch_nac = compute_nac(pred, y)
        batch_metrics["rmse"].append(batch_rmse)
        batch_metrics["acc"].append(batch_acc)
        batch_metrics["nac"].append(batch_nac)

        # 每batch保存 checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_acc": best_acc,
            "train_loss_list": train_loss_list,
            "val_loss_list": val_loss_list,
            "val_rmse_list": val_rmse_list,
            "val_acc_list": val_acc_list,
            "val_nac_list": val_nac_list,
            "batch_metrics": batch_metrics
        }, CKPT_LAST)

    train_loss_list.append(total_loss / len(train_loader))

    # ---------------- Val ----------------
    model.eval()
    total_val_loss, total_rmse, total_acc, total_nac = 0, 0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Val"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            total_val_loss += loss_fn(pred, y).item()
            total_rmse += compute_rmse(pred, y)
            total_acc += compute_acc(pred, y)
            total_nac += compute_nac(pred, y)

    val_loss = total_val_loss / len(val_loader)
    val_rmse = total_rmse / len(val_loader)
    val_acc = total_acc / len(val_loader)
    val_nac = total_nac / len(val_loader)

    val_loss_list.append(val_loss)
    val_rmse_list.append(val_rmse)
    val_acc_list.append(val_acc)
    val_nac_list.append(val_nac)

    print(f"Epoch {epoch+1}: Train={train_loss_list[-1]:.4f} | "
          f"Val={val_loss:.4f} | RMSE={val_rmse:.4f} | ACC={val_acc:.4f} | NAC={val_nac:.4f}")

    # 保存 best 模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "best_acc": best_acc
        }, CKPT_BEST)
        print(f"[Best Model] Epoch {epoch+1}  ACC={best_acc:.4f}")

# ===========================
# 作图
# ===========================
plt.figure(figsize=(12,8))

plt.subplot(4,1,1)
plt.plot(train_loss_list, label="Train Loss")
plt.plot(val_loss_list, label="Val Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(4,1,2)
plt.plot(val_rmse_list, label="Val RMSE")
plt.legend()
plt.title("Validation RMSE")

plt.subplot(4,1,3)
plt.plot(val_acc_list, label="Val ACC")
plt.legend()
plt.title("Validation ACC")

plt.subplot(4,1,4)
plt.plot(val_nac_list, label="Val NAC")
plt.legend()
plt.title("Validation NAC")

plt.tight_layout()
plt.show()
