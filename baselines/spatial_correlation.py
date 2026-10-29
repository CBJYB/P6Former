import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.config import DATA_DIR, SPATIAL_SIZE, TIME_STEPS, VARIABLES, NUM_INDICES, D_MODEL, N_HEADS, NUM_LAYERS, D_FF, DROPOUT
from model.dataset import P6FormerDataset
from model.model import P6Former

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INDICES_NAMES = ['txx', 'tnn', 'cdd', 'cwd', 'r95p', 'sdii']
CKPT_BASE = r"D:\Pycharm\save\Plateau-disaster-risk-assessment\checkpoints_7d"

X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
N = y_test.shape[0]

def spatial_correlation(pred, true):
    N, C, H, W = pred.shape
    corr_list = [[] for _ in range(C)]
    for i in range(N):
        for c in range(C):
            p = pred[i, c].flatten()
            t = true[i, c].flatten()
            valid = ~(np.isnan(p) | np.isnan(t))
            if valid.sum() < 10: continue
            p, t = p[valid], t[valid]
            if p.std() < 1e-8:
                corr_list[c].append(0.0)
            else:
                r, _ = pearsonr(p, t)
                if not np.isnan(r): corr_list[c].append(r)
    return np.array([np.mean(c) if c else 0.0 for c in corr_list])


# P6Former 评估
print("\n🔍 P6Former...")
test_dataset = P6FormerDataset(x_file=os.path.join(DATA_DIR, "X_test.npy"), y_file=os.path.join(DATA_DIR, "y_test.npy"), indices_names=INDICES_NAMES, target_spatial=SPATIAL_SIZE, augment=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

model_path = os.path.join(CKPT_BASE, "lam1e-1_seed42", "best_model.pth")
summary_path = os.path.join(CKPT_BASE, "lam1e-1_seed42", "summary.npy")

if os.path.exists(model_path) and os.path.exists(summary_path):
    summary = np.load(summary_path, allow_pickle=True).item()
    var_mean, var_std = summary['var_mean'], summary['var_std']

    model = P6Former(time_steps=TIME_STEPS, in_vars=VARIABLES, num_indices=NUM_INDICES, d_model=D_MODEL, n_heads=N_HEADS, num_layers=NUM_LAYERS, d_ff=D_FF, dropout=DROPOUT, spatial_size=SPATIAL_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    preds = []
    with torch.no_grad():
        for X, _ in tqdm(test_loader, desc="P6Former predicting"):
            m = torch.tensor(var_mean, dtype=torch.float32, device=DEVICE).view(1, 1, -1, 1, 1)
            s = torch.tensor(var_std, dtype=torch.float32, device=DEVICE).view(1, 1, -1, 1, 1)
            X = (X.to(DEVICE) - m) / (s + 1e-8)
            pred_y, _ = model(X)
            preds.append(pred_y.cpu().numpy())

    pred_p6 = np.concatenate(preds, axis=0)
    corr_p6 = spatial_correlation(pred_p6, y_test)
    print(f"   Spatial correlation: {corr_p6.round(4)}")
else:
    print("❌ 未找到 P6Former 模型或 summary.npy")
    corr_p6 = np.zeros(6)

# ConvLSTM 评估
print("\n🔍 ConvLSTM...")
