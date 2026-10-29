
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd

# 🔧 核心修复：将项目根目录加入环境变量
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.config import (TIME_STEPS, VARIABLES, NUM_INDICES, D_MODEL, N_HEADS, NUM_LAYERS, D_FF, DROPOUT, SPATIAL_SIZE,
                          DATA_DIR)
from model.dataset import P6FormerDataset
from model.model import P6Former
from model.physics_loss import PhysicsLoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_BASE = r"D:\Pycharm\save\Plateau-disaster-risk-assessment\checkpoints_7d"
OUTPUT_DIR = "paper_assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_ds = P6FormerDataset(x_file=os.path.join(DATA_DIR, "X_test.npy"), y_file=os.path.join(DATA_DIR, "y_test.npy"),
                          indices_names=['txx', 'tnn', 'cdd', 'cwd', 'r95p', 'sdii'], target_spatial=SPATIAL_SIZE,
                          augment=False)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)


def normalize(x, var_mean, var_std):
    m = torch.tensor(var_mean, dtype=torch.float32, device=x.device).view(1, 1, -1, 1, 1)
    s = torch.tensor(var_std, dtype=torch.float32, device=x.device).view(1, 1, -1, 1, 1)
    return (x - m) / (s + 1e-8)


def load_model_and_params(exp_name):
    model_path = os.path.join(CHECKPOINT_BASE, exp_name, "best_model.pth")
    summary_path = os.path.join(CHECKPOINT_BASE, exp_name, "summary.npy")
    if not os.path.exists(model_path) or not os.path.exists(summary_path):
        print(f"⚠️ 找不到模型或 summary: {exp_name}")
        return None, None, None
    summary = np.load(summary_path, allow_pickle=True).item()
    if 'var_mean' not in summary:
        print(f"⚠️ {exp_name} 的 summary 里没有 var_mean")
        return None, None, None

    model = P6Former(time_steps=TIME_STEPS, in_vars=VARIABLES, num_indices=NUM_INDICES, d_model=D_MODEL,
                     n_heads=N_HEADS, num_layers=NUM_LAYERS, d_ff=D_FF, dropout=DROPOUT, spatial_size=SPATIAL_SIZE).to(
        DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model, summary['var_mean'], summary['var_std']


results = {}
for exp_name, label in [("lam1e-1_seed42", "P6Former"), ("seed42_nophys", "w/o Physics")]:
    print(f"\n🔍 Evaluating: {label} ({exp_name})")
    model, var_mean, var_std = load_model_and_params(exp_name)
    if model is None: continue

    physics_loss = PhysicsLoss(var_mean=var_mean, var_std=var_std).to(DEVICE)
    all_res = {k: [] for k in ['cont', 'thermo', 'mom', 'energy', 'phase', 'covar']}

    with torch.no_grad():
        for X, _ in test_loader:
            X = normalize(X, var_mean, var_std).to(DEVICE)
            _, aux_out = model(X)
            res = physics_loss(aux_out)
            for k in all_res:
                all_res[k].append(res[k].item())

    results[label] = {k: np.mean(v) for k, v in all_res.items()}
    print(f"   Average residuals (True Physical Units):")
    for k, v in results[label].items():
        print(f"      {k:>8}: {v:.6f}")

if 'P6Former' in results and 'w/o Physics' in results:
    table_data = []
    print("\n" + "=" * 70)
    print(f"{'Constraint':<12} {'P6Former':>15} {'w/o Physics':>15} {'Reduction':>12}")
    print("-" * 70)
    for c in ['cont', 'thermo', 'mom', 'energy', 'phase', 'covar']:
        p6_v = results['P6Former'][c]
        no_v = results['w/o Physics'][c]
        red = (no_v - p6_v) / no_v * 100 if no_v > 1e-8 else 100.0
        table_data.append({'Constraint': c, 'P6Former': p6_v, 'w/o Physics': no_v, 'Reduction (%)': red})
        print(f"{c:<12} {p6_v:>15.6f} {no_v:>15.6f} {red:>11.1f}%")

    pd.DataFrame(table_data).to_csv(os.path.join(OUTPUT_DIR, "table_physics_satisfaction.csv"), index=False)
    print("=" * 70)
    print(f"\n✅ 生成真实物理残差表: {os.path.join(OUTPUT_DIR, 'table_physics_satisfaction.csv')}")