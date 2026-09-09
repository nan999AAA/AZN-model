# MSLP-3 TC-subseasonal_model_mslp_TransformerCNN-2
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

SAVE_DIR = "./TransformerCNN-2"
os.makedirs(SAVE_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(SAVE_DIR, "last_checkpoint-2.pth")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "subseasonal_model_mslp_TransformerCNN-2.pth")

# ===========================
# 输入输出变量
# ===========================
input_vars = (
    [f'mslp_hist_{i}' for i in range(20)] +
    [f'z850_hist_{i}' for i in range(10)] +
    [f'z500_z850_hist_{i}' for i in range(10)] +
    [f'q700_hist_{i}' for i in range(10)] +
    [f'divergence900_hist_{i}' for i in range(10)] +
    [f'pv900_hist_{i}' for i in range(10)] +
    ['pred_msl_month', 'pred_t2m_month', 'elevation']
)

target_vars = [
    'mslp_mon', 'mslp_wed', 'mslp_fri', 'mslp_sun',
    'mslp_tue_next', 'mslp_thu_next', 'mslp_sat_next'
]

# ===========================
# 加载数据
# ===========================
ds = xr.open_dataset(DATA_PATH)
for v in input_vars + target_vars:
    ds[v] = ds[v].fillna(0)

# ===========================
# 数据集
# ===========================
class SubseasonalDataset(Dataset):
    def __init__(self, ds, input_vars, target_vars):
        self.x = ds[input_vars].to_array().transpose(
            'time', 'variable', 'latitude', 'longitude'
        )
        self.y = ds[target_vars].to_array().transpose(
            'time', 'variable', 'latitude', 'longitude'
        )

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx].values, dtype=torch.float32),
            torch.tensor(self.y[idx].values, dtype=torch.float32)
        )

dataset = SubseasonalDataset(ds, input_vars, target_vars)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# ===========================
# Transformer + CNN
# ===========================
class TransformerCNN(nn.Module):
    def __init__(self, in_channels, out_channels, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.patch = nn.Conv2d(in_channels, d_model, 3, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 1)
        )

    def forward(self, x):
        x = self.patch(x)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)   # (B, H*W, C)
        x = self.transformer(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return self.decoder(x)

sample_x, _ = dataset[0]
model = TransformerCNN(len(input_vars), len(target_vars)).to(DEVICE)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===========================
# 指标（严格按你的定义）
# ===========================
def compute_rmse(pred, truth):
    return torch.sqrt(torch.mean((pred - truth) ** 2)).item()

def compute_acc(pred, truth):
    pred_flat = pred.view(-1)
    truth_flat = truth.view(-1)

    pred_anom = pred_flat - pred_flat.mean()
    truth_anom = truth_flat - truth_flat.mean()

    denom = torch.sqrt(
        torch.sum(pred_anom ** 2) * torch.sum(truth_anom ** 2)
    )

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
# 断点恢复
# ===========================
start_epoch = 0
start_batch = 0

train_loss_list, val_loss_list = [], []
rmse_epoch_list, acc_epoch_list, nac_epoch_list = [], [], []

batch_metrics = {
    "rmse": [],
    "acc": [],
    "nac": []
}

best_acc = -np.inf

if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"]
    start_batch = ckpt["batch"]

    train_loss_list = ckpt["train_loss_list"]
    val_loss_list = ckpt["val_loss_list"]
    rmse_epoch_list = ckpt["rmse_epoch_list"]
    acc_epoch_list = ckpt["acc_epoch_list"]
    nac_epoch_list = ckpt["nac_epoch_list"]
    batch_metrics = ckpt["batch_metrics"]
    best_acc = ckpt["best_acc"]

    print(f"🔁 从 epoch {start_epoch}, batch {start_batch} 恢复")

# ===========================
# 训练
# ===========================
for epoch in range(start_epoch, EPOCHS):
    model.train()
    epoch_loss = 0.0

    for b, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        if epoch == start_epoch and b < start_batch:
            continue

        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # ===== batch 指标 =====
        rmse_b = compute_rmse(pred, y)
        acc_b = compute_acc(pred, y)
        nac_b = compute_nac(pred, y)

        batch_metrics["rmse"].append(rmse_b)
        batch_metrics["acc"].append(acc_b)
        batch_metrics["nac"].append(nac_b)

        # ===== batch 断点 =====
        torch.save({
            "epoch": epoch,
            "batch": b + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss_list": train_loss_list,
            "val_loss_list": val_loss_list,
            "rmse_epoch_list": rmse_epoch_list,
            "acc_epoch_list": acc_epoch_list,
            "nac_epoch_list": nac_epoch_list,
            "batch_metrics": batch_metrics,
            "best_acc": best_acc
        }, CHECKPOINT_PATH)

    train_loss_list.append(epoch_loss / len(train_loader))

    # =======================
    # 验证
    # =======================
    model.eval()
    rmse_v, acc_v, nac_v, loss_v = 0, 0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)

            loss_v += loss_fn(pred, y).item()
            rmse_v += compute_rmse(pred, y)
            acc_v += compute_acc(pred, y)
            nac_v += compute_nac(pred, y)

    rmse_v /= len(val_loader)
    acc_v /= len(val_loader)
    nac_v /= len(val_loader)
    loss_v /= len(val_loader)

    val_loss_list.append(loss_v)
    rmse_epoch_list.append(rmse_v)
    acc_epoch_list.append(acc_v)
    nac_epoch_list.append(nac_v)

    print(
        f"Epoch {epoch+1:03d} | "
        f"Val RMSE={rmse_v:.4f} | "
        f"ACC={acc_v:.4f} | "
        f"NAC={nac_v:.4f}"
    )

    # =======================
    # 保存 ACC 最优模型
    # =======================
    if acc_v > best_acc:
        best_acc = acc_v
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"✅ 保存最佳模型（ACC={best_acc:.4f}）")

