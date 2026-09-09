# PR-SA-33best_ACC_model

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

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_DIR = SCRIPT_DIR.parent

# ===========================
# 配置
# ===========================

DATA_PATH = PROJECT_DIR / "tp_2D.nc"
DEVICE = torch.device("cpu")
BATCH_SIZE = 1
EPOCHS = 500
LEARNING_RATE = 1e-3

SAVE_DIR = "./33SpatialAttentionResNet_TP"
os.makedirs(SAVE_DIR, exist_ok=True)

CKPT_LAST = os.path.join(SAVE_DIR, "33last_checkpoint.pth")
CKPT_BEST_ACC = os.path.join(SAVE_DIR, "33best_ACC_model.pth")
METRIC_CSV = os.path.join(SAVE_DIR, "33training_metrics.csv")

# ===========================
# 输入输出变量
# ===========================
input_vars = (
    [f'tp_hist_{i}' for i in range(20)] +
    [f'gh200_hist_{i}' for i in range(10)] +
    [f'gh200_300_hist_{i}' for i in range(10)] +
    [f'gh200_500_hist_{i}' for i in range(10)] +
    [f'q700_hist_{i}' for i in range(10)] +
    [f'cloudfraction800_hist_{i}' for i in range(10)] +
    ['pred_tp_month', 'elevation']
)

target_vars = [
    'tp_mon', 'tp_wed', 'tp_fri', 'tp_sun',
    'tp_tue_next', 'tp_thu_next', 'tp_sat_next'
]

# ===========================
# 数据加载
# ===========================
ds = xr.open_dataset(DATA_PATH)
for var in input_vars + target_vars:
    if ds[var].isnull().any():
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
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# ===========================
# 模型定义
# ===========================
VAR_GROUPS = {
    "tp": list(range(0, 20)),
    "gh": list(range(20, 50)),
    "q":  list(range(50, 60)),
    "cloud": list(range(60, 70)),
    "static": list(range(70, len(input_vars)))
}

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.attn = SpatialAttention()
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.attn(out)
        return self.relu(out + self.skip(x))

class VariableEncoder(nn.Module):
    def __init__(self, in_c, out_c=32):
        super().__init__()
        self.block = nn.Sequential(
            ResidualBlock(in_c, out_c),
            ResidualBlock(out_c, out_c)
        )

    def forward(self, x):
        return self.block(x)

class TimeStepDecoder(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.decoder = nn.Sequential(
            ResidualBlock(in_c, 64),
            ResidualBlock(64, 32),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        return self.decoder(x)

class SpatialAttentionResNet(nn.Module):
    def __init__(self, input_vars, target_vars):
        super().__init__()

        self.encoders = nn.ModuleDict({
            name: VariableEncoder(len(idxs))
            for name, idxs in VAR_GROUPS.items()
        })

        fusion_c = 32 * len(VAR_GROUPS)

        self.fusion = nn.Sequential(
            ResidualBlock(fusion_c, 128),
            ResidualBlock(128, 64)
        )

        self.decoders = nn.ModuleDict({
            name: TimeStepDecoder(64)
            for name in target_vars
        })

    def forward(self, x):
        feats = []
        for name, idxs in VAR_GROUPS.items():
            feats.append(self.encoders[name](x[:, idxs]))

        x = torch.cat(feats, dim=1)
        x = self.fusion(x)

        outputs = []
        for name in self.decoders:
            outputs.append(self.decoders[name](x))

        return torch.cat(outputs, dim=1)

model = SpatialAttentionResNet(input_vars, target_vars).to(DEVICE)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===========================
# 指标函数
# ===========================
def compute_rmse(pred, truth):
    return torch.sqrt(torch.mean((pred - truth) ** 2)).item()

def compute_acc(pred, truth):
    p = pred.view(-1) - pred.mean()
    t = truth.view(-1) - truth.mean()
    denom = torch.sqrt(torch.sum(p**2) * torch.sum(t**2))
    return 0.0 if denom == 0 else (torch.sum(p * t) / denom).item()

def compute_nac(pred, truth):
    rmse = compute_rmse(pred, truth)
    std = torch.std(truth)
    return 0.0 if std == 0 else 1 - rmse / std.item()

# ===========================
# 断点恢复
# ===========================
start_epoch = 0
best_acc = -1e9

train_loss_list, val_loss_list = [], []
val_rmse_list, val_acc_list, val_nac_list = [], [], []

if os.path.exists(CKPT_LAST):
    ckpt = torch.load(CKPT_LAST, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"] + 1
    best_acc = ckpt["best_acc"]

    train_loss_list = ckpt["train_loss"]
    val_loss_list = ckpt["val_loss"]
    val_rmse_list = ckpt["rmse"]
    val_acc_list = ckpt["acc"]
    val_nac_list = ckpt["nac"]

    print(f"✅ 从断点恢复训练：epoch {start_epoch}")
else:
    print("🆕 未检测到断点，从头开始训练")
    with open(METRIC_CSV, "w") as f:
        f.write("epoch,train_loss,val_loss,rmse,acc,nac\n")

# ===========================
# 训练
# ===========================
for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Train"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    train_loss_list.append(train_loss)

    model.eval()
    vl = rmse = acc = nac = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            vl += loss_fn(pred, y).item()
            rmse += compute_rmse(pred, y)
            acc += compute_acc(pred, y)
            nac += compute_nac(pred, y)

    vl /= len(val_loader)
    rmse /= len(val_loader)
    acc /= len(val_loader)
    nac /= len(val_loader)

    val_loss_list.append(vl)
    val_rmse_list.append(rmse)
    val_acc_list.append(acc)
    val_nac_list.append(nac)

    print(f"Epoch {epoch+1}: "
          f"TrainLoss={train_loss:.4f} | ValLoss={vl:.4f} | "
          f"RMSE={rmse:.4f} | ACC={acc:.4f} | NAC={nac:.4f}")

    with open(METRIC_CSV, "a") as f:
        f.write(f"{epoch+1},{train_loss:.6f},{vl:.6f},{rmse:.6f},{acc:.6f},{nac:.6f}\n")

    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_acc": best_acc,
        "train_loss": train_loss_list,
        "val_loss": val_loss_list,
        "rmse": val_rmse_list,
        "acc": val_acc_list,
        "nac": val_nac_list
    }, CKPT_LAST)

    if acc > best_acc:
        best_acc = acc
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "best_acc": best_acc
        }, CKPT_BEST_ACC)
        print(f"🏆 Best ACC 更新：Epoch {epoch+1}, ACC={best_acc:.4f}")

print("✅ 训练完成（支持断点续训）")

