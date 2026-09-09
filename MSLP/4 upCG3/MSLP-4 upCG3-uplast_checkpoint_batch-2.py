# MSLP-4 upCG3-uplast_checkpoint_batch-2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# ===========================
# 指标函数
# ===========================
def compute_rmse(pred, truth):
    return torch.sqrt(torch.mean((pred - truth) ** 2)).item()

def compute_acc(pred, truth):
    """
    原始 ACC：把所有维度拉平，算一对一相关
    """
    pred_flat = pred.view(-1)
    truth_flat = truth.view(-1)
    pred_anom = pred_flat - pred_flat.mean()
    truth_anom = truth_flat - truth_flat.mean()
    denom = torch.sqrt(torch.sum(pred_anom**2) * torch.sum(truth_anom**2))
    if denom == 0:
        return 0.0
    return (torch.sum(pred_anom * truth_anom) / denom).item()

def compute_acc_7day_mean(pred, truth):
    """
    新 ACC：每个预报日单独算空间 ACC，再对 7 天平均
    pred, truth: [B, 7, lat, lon] 或 [7, lat, lon]
    """
    # 如果有 batch 维度，先去掉
    if pred.dim() == 4:
        pred = pred.squeeze(0)
        truth = truth.squeeze(0)

    acc_days = []
    for d in range(pred.shape[0]):  # 遍历 7 天
        acc_d = compute_acc(pred[d], truth[d])  # 只在空间维度算
        acc_days.append(acc_d)

    return float(np.mean(acc_days))

def compute_nac(pred, truth):
    rmse = compute_rmse(pred, truth)
    truth_std = torch.std(truth)
    if truth_std == 0:
        return 0.0
    return 1 - (rmse / truth_std).item()

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_DIR = SCRIPT_DIR.parent

# ===========================
# 配置
# ===========================

DATA_PATH = PROJECT_DIR / "mslp_2D.nc"
DEVICE = torch.device("cpu")
BATCH_SIZE = 1
EPOCHS = 450
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.1
CHECKPOINT_DIR = "./upConvGRUCell-2"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "upbest_ACC_model-2.pth")
LAST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "uplast_checkpoint_batch-2.pth")

# ===========================
# 输入输出变量
# ===========================
input_vars = (
    [f"mslp_hist_{i}" for i in range(20)] +
    [f"z850_hist_{i}" for i in range(10)] +
    [f"z500_z850_hist_{i}" for i in range(10)] +
    [f"q700_hist_{i}" for i in range(10)] +
    [f"divergence900_hist_{i}" for i in range(10)] +
    [f"pv900_hist_{i}" for i in range(10)] +
    ["pred_msl_month", "pred_t2m_month", "elevation"]
)

target_vars = [
    "mslp_mon", "mslp_wed", "mslp_fri",
    "mslp_sun", "mslp_tue_next",
    "mslp_thu_next", "mslp_sat_next"
]

# ===========================
# 数据读取 & 标准化
# ===========================
ds = xr.open_dataset(DATA_PATH)

for v in input_vars + target_vars:
    ds[v] = ds[v].fillna(0)

for v in input_vars:
    mean = ds[v].mean()
    std = ds[v].std()
    ds[v] = (ds[v] - mean) / (std if std != 0 else 1)

# ===========================
# Dataset
# ===========================
class SubseasonalDataset(Dataset):
    def __init__(self, ds):
        self.x = ds[input_vars].to_array().transpose("time", "variable", "latitude", "longitude")
        self.y = ds[target_vars].to_array().transpose("time", "variable", "latitude", "longitude")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx].values, dtype=torch.float32),
            torch.tensor(self.y[idx].values, dtype=torch.float32)
        )

dataset = SubseasonalDataset(ds)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# ===========================
# 改进版 ConvGRUCell
# ===========================
class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 3 * hidden_dim, kernel_size, padding=padding)
        self.norm = nn.GroupNorm(3, 3 * hidden_dim)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.norm(self.conv(combined))
        z, r, h_hat = torch.chunk(gates, 3, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        h_hat = torch.tanh(r * h_hat)
        return (1 - z) * h_prev + z * h_hat

# ===========================
# 改进版 ConvGRUNet
# ===========================
class ConvGRUNet(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, gru_steps=3):
        super().__init__()
        self.gru_steps = gru_steps
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_ch, hid_ch, 3, padding=1)
        )
        self.encoder_norm = nn.GroupNorm(8, hid_ch)
        self.rnn = ConvGRUCell(hid_ch, hid_ch, 3)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.decoder = nn.Conv2d(hid_ch, out_ch, 1)

    def forward(self, x):
        feat = self.encoder_norm(self.encoder(x))
        h = torch.zeros_like(feat)
        for _ in range(self.gru_steps):
            h = self.rnn(feat, h)
        h = h + feat
        h = self.dropout(h)
        return self.decoder(h)

# ===========================
# 模型实例化
# ===========================
model = ConvGRUNet(len(input_vars), 64, len(target_vars), gru_steps=3).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# ===========================
# 断点恢复
# ===========================
start_epoch, start_batch = 0, 0
best_acc = -np.inf
history = {
    "rmse": [], "acc": [], "nac": [],
    "batch_rmse": [], "batch_acc": [], "batch_nac": []
}

if os.path.exists(LAST_CHECKPOINT_PATH):
    try:
        ckpt = torch.load(LAST_CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt.get("optimizer", optimizer.state_dict()))
        start_epoch = ckpt.get("epoch", 0)
        start_batch = ckpt.get("batch", 0)
        best_acc = ckpt.get("best_acc", best_acc)
        history = ckpt.get("history", history)
        print(f"恢复训练：Epoch={start_epoch}, Batch={start_batch}, Best ACC={best_acc:.4f}")
    except Exception as e:
        print("Checkpoint 加载失败，重新从零开始训练:", e)

# ===========================
# 训练主循环
# ===========================
for epoch in range(start_epoch, EPOCHS):
    model.train()
    for bidx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        if epoch == start_epoch and bidx < start_batch:
            continue
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        # ===== batch 级指标 =====
        batch_rmse = compute_rmse(pred, y)
        batch_acc = compute_acc_7day_mean(pred, y)  # 改这里
        batch_nac = compute_nac(pred, y)
        history["batch_rmse"].append(batch_rmse)
        history["batch_acc"].append(batch_acc)
        history["batch_nac"].append(batch_nac)

        # batch 级断点保存
        torch.save({
            "epoch": epoch,
            "batch": bidx + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
            "history": history
        }, LAST_CHECKPOINT_PATH)

    # ===== 验证（每轮）=====
    model.eval()
    rmse, acc, nac = 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x.to(DEVICE))
            rmse += compute_rmse(pred, y)
            acc += compute_acc_7day_mean(pred, y)  # 改这里
            nac += compute_nac(pred, y)
    rmse /= len(val_loader)
    acc /= len(val_loader)
    nac /= len(val_loader)

    history["rmse"].append(rmse)
    history["acc"].append(acc)
    history["nac"].append(nac)
    print(f"Epoch {epoch+1}: RMSE={rmse:.4f} | ACC={acc:.4f} | NAC={nac:.4f}")

    # 保存最佳模型
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"✓ 保存最佳模型 ACC={best_acc:.4f}")

