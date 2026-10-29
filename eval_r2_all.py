import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

from model.config import (DATA_DIR, SPATIAL_SIZE, TIME_STEPS, VARIABLES,
                          NUM_INDICES, D_MODEL, N_HEADS, NUM_LAYERS, D_FF, DROPOUT)
from model.dataset import P6FormerDataset
from model.model import P6Former

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_7d")
OUT_CSV = os.path.join(BASE_DIR, "paper_assets", "p6former_full_eval.csv")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

INDEX_NAMES = ['TXx', 'TNn', 'CDD', 'CWD', 'R95p', 'SDII']
SEEDS = [42, 123, 2024]

RUNS = {
    "phys": {s: f"lam1e-1_seed{s}" for s in SEEDS},
    "nophys": {42: "seed42_nophys", 123: "seed123_nophys", 2024: "seed2024_nophys"},
    "noaug": {s: f"lam1e-1_noaug_seed{s}" for s in SEEDS},
    "baseline": {s: f"seed{s}_baseline" for s in SEEDS},
}

# 加载测试集
test_ds = P6FormerDataset(
    x_file=os.path.join(DATA_DIR, "X_test.npy"),
    y_file=os.path.join(DATA_DIR, "y_test.npy"),
    indices_names=['txx', 'tnn', 'cdd', 'cwd', 'r95p', 'sdii'],
    target_spatial=SPATIAL_SIZE, augment=False
)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)


def normalize(x, var_mean, var_std):
    """逐变量归一化"""
    m = torch.tensor(var_mean, dtype=torch.float32, device=x.device).view(1, 1, -1, 1, 1)
    s = torch.tensor(var_std, dtype=torch.float32, device=x.device).view(1, 1, -1, 1, 1)
    return (x - m) / (s + 1e-8)


def evaluate(model_path, current_num_layers):
    model = P6Former(
        time_steps=TIME_STEPS, in_vars=VARIABLES, num_indices=NUM_INDICES,
        d_model=D_MODEL, n_heads=N_HEADS, num_layers=current_num_layers,
        d_ff=D_FF, dropout=DROPOUT, spatial_size=SPATIAL_SIZE
    ).to(DEVICE)

    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except RuntimeError as e:
        print(f"   ⚠️ 权重尺寸不匹配，跳过: {os.path.basename(os.path.dirname(model_path))}")
        return [float('nan')] * 6

    model.eval()

    summary_path = os.path.join(os.path.dirname(model_path), "summary.npy")
    if not os.path.exists(summary_path):
        return [float('nan')] * 6
    summary = np.load(summary_path, allow_pickle=True).item()
    if 'var_mean' not in summary:
        return [float('nan')] * 6
    var_mean, var_std = summary['var_mean'], summary['var_std']

    preds, trues = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = normalize(X, var_mean, var_std).to(DEVICE)
            pred, _ = model(X)
            preds.append(pred.cpu().numpy())
            trues.append(y.numpy())

    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    pred_avg = pred.mean(axis=(2, 3))
    true_avg = true.mean(axis=(2, 3))

    if np.isnan(pred_avg).any():
        return [float('nan')] * 6

    return [float(r2_score(true_avg[:, i], pred_avg[:, i])) for i in range(6)]


rows = []
print("=" * 60)
print("🚀 开始批量评估 P6Former 的所有变体")
print("=" * 60)

for cfg, seeds in RUNS.items():
    current_L = NUM_LAYERS  # 统一使用 3 层
    for seed, dirname in seeds.items():
        model_path = os.path.join(CKPT_DIR, dirname, "best_model.pth")

        if not os.path.exists(model_path):
            print(f"⏭️ 缺失: {dirname}，跳过")
            continue

        r2s = evaluate(model_path, current_L)

        if np.isnan(r2s).any():
            continue

        rows.append({
            "Config": f"seed{seed}_{cfg}",
            **{f"{idx}_R2": v for idx, v in zip(INDEX_NAMES, r2s)},
            "Mean_R2": float(np.nanmean(r2s))
        })
        print(f"🔍 {dirname:<25} Mean R² = {np.nanmean(r2s):.4f}")

# 写入 CSV
with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=["Config"] + [f"{idx}_R2" for idx in INDEX_NAMES] + ["Mean_R2"])
    w.writeheader()
    w.writerows(rows)

print("=" * 60)
print(f"✅ 成功写入 {len(rows)} 组评估结果至: {OUT_CSV}")
print("=" * 60)