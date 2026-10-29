
import os
import csv
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from sklearn.metrics import r2_score

from model.config import DATA_DIR, SPATIAL_SIZE, TIME_STEPS, VARIABLES, NUM_INDICES, D_MODEL, N_HEADS, NUM_LAYERS, D_FF, DROPOUT
from model.dataset import P6FormerDataset
from model.model import P6Former

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

matplotlib.rcParams.update({
    'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'figure.titlesize': 18, 'font.family': 'DejaVu Sans', 'axes.unicode_minus': False
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "paper_assets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_7d")
CONVLSTM_DIR = os.path.join(BASE_DIR, "data")
INDEX_NAMES = ['TXx', 'TNn', 'CDD', 'CWD', 'R95p', 'SDII']
SEEDS = [42, 123, 2024]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_real_inference(model_path):
    print(f"  ⏳ 加载模型进行真实推理: {os.path.basename(os.path.dirname(model_path))}")
    test_ds = P6FormerDataset(
        x_file=os.path.join(DATA_DIR, "X_test.npy"),
        y_file=os.path.join(DATA_DIR, "y_test.npy"),
        indices_names=['txx', 'tnn', 'cdd', 'cwd', 'r95p', 'sdii'],
        target_spatial=SPATIAL_SIZE, augment=False
    )
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    summary_path = os.path.join(os.path.dirname(model_path), "summary.npy")
    summary = np.load(summary_path, allow_pickle=True).item()
    var_mean = torch.tensor(summary['var_mean'], dtype=torch.float32, device=DEVICE).view(1, 1, -1, 1, 1)
    var_std = torch.tensor(summary['var_std'], dtype=torch.float32, device=DEVICE).view(1, 1, -1, 1, 1)

    model = P6Former(
        time_steps=TIME_STEPS, in_vars=VARIABLES, num_indices=NUM_INDICES,
        d_model=D_MODEL, n_heads=N_HEADS, num_layers=NUM_LAYERS,
        d_ff=D_FF, dropout=DROPOUT, spatial_size=SPATIAL_SIZE
    ).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    preds_maps, true_maps = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = ((X.to(DEVICE) - var_mean) / (var_std + 1e-8))
            pred, _ = model(X)
            preds_maps.append(pred.cpu().numpy())
            true_maps.append(y.numpy())

    pred_maps = np.concatenate(preds_maps, axis=0)
    true_maps = np.concatenate(true_maps, axis=0)
    return true_maps.mean(axis=(2, 3)), pred_maps.mean(axis=(2, 3)), true_maps, pred_maps

# 加载动态数据
print("=" * 70 + "\n加载动态数据...\n" + "=" * 70)

csv_path = os.path.join(OUTPUT_DIR, "p6former_full_eval.csv")
if os.path.exists(csv_path):
    df_eval = pd.read_csv(csv_path)
    def get_agg_r2(cfg_name):
        sub = df_eval[df_eval['Config'].str.endswith(cfg_name)]
        if sub.empty: return float('nan'), float('nan')
        return float(sub['Mean_R2'].mean()), float(sub['Mean_R2'].std(ddof=0))

    phys_mean, phys_std = get_agg_r2('_phys')
    nophys_mean, nophys_std = get_agg_r2('_nophys')
    noaug_mean, noaug_std = get_agg_r2('_noaug')
    baseline_mean, baseline_std = get_agg_r2('_baseline')
else:
    print("⚠️ 找不到 p6former_full_eval.csv，R2 相关图表将跳过。")
    df_eval = None

# 生成表格数据
print("\n" + "=" * 70 + "\n生成表格数据...\n" + "=" * 70)

# 物理残差数据
res_src = os.path.join(OUTPUT_DIR, "table_physics_satisfaction.csv")
if os.path.exists(res_src):
    shutil.copy(res_src, os.path.join(OUTPUT_DIR, "table_residuals_export.csv"))
else:
    print("⚠️ 未找到 table_physics_satisfaction.csv，请先运行 physics_satisfaction.py")

# 消融实验数据
if df_eval is not None:
    ablation_r2_data = {
        'Configuration': ['P6Former (Full)', 'w/o Physics', 'w/o Augmentation', 'Baseline'],
        'Avg R2': [phys_mean, nophys_mean, noaug_mean, baseline_mean],
        'Std': [phys_std, nophys_std, noaug_std, baseline_std],
    }
    pd.DataFrame(ablation_r2_data).to_csv(os.path.join(OUTPUT_DIR, "table_ablation_export.csv"), index=False)

# 生成图表
print("\n" + "=" * 70 + "\n生成图表...\n" + "=" * 70)

model_path = os.path.join(CKPT_DIR, "lam1e-1_seed42", "best_model.pth")
if os.path.exists(model_path):
    real_true_avg, real_pred_avg, real_true_maps, real_pred_maps = run_real_inference(model_path)
else:
    print("⚠️ 找不到 lam1e-1_seed42 模型，跳过图3/图6/图8。")
    real_true_avg = real_pred_avg = real_true_maps = real_pred_maps = None

# 图 2：研究区域
if HAS_CARTOPY:
    try:
        fig = plt.figure(figsize=(10, 8)); ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([73, 108, 22, 42], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#F5F5DC'); ax.add_feature(cfeature.OCEAN, facecolor='#E0F3F8')
        ax.add_feature(cfeature.RIVERS, linewidth=0.5, color='royalblue', alpha=0.7)
        ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='royalblue', alpha=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1, color='gray'); ax.add_feature(cfeature.COASTLINE, linewidth=1, color='black')
        rect = Rectangle((80, 25), 25, 15, fill=False, edgecolor='red', linewidth=2, linestyle='--'); ax.add_patch(rect)
        cities = {'Lhasa': (91.1, 29.6), 'Nagqu': (92.0, 31.4), 'Golmud': (94.9, 36.4), 'Xining': (101.8, 36.6), 'Shigatse': (88.8, 29.2), 'Hotan': (79.9, 37.1)}
        for name, (lon, lat) in cities.items():
            ax.plot(lon, lat, 'ro', markersize=6, transform=ccrs.PlateCarree())
            ax.text(lon + 0.5, lat + 0.3, name, fontsize=9, fontweight='bold', transform=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.4, linestyle='--', color='gray')
        gl.top_labels = False; gl.right_labels = False
        ax.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.85), arrowprops=dict(facecolor='black', width=3, headwidth=8), ha='center', va='center', fontsize=14, xycoords='axes fraction')
        ax.set_title('Study Area: Tibetan Plateau', fontsize=15, fontweight='bold', pad=15)
        plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "fig2_study_area.png"), dpi=300, bbox_inches='tight'); plt.close()
    except Exception as e:
        print(f"⚠️ Cartopy 渲染失败: {e}")

# 图 3：真实预测 vs 真实散点图
if real_true_avg is not None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10)); axes = axes.flatten()
    for i, idx_name in enumerate(INDEX_NAMES):
        ax = axes[i]; t, p = real_true_avg[:, i], real_pred_avg[:, i]
        ax.scatter(t, p, alpha=0.5, s=15, c='#2E86AB')
        mn, mx = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1)
        ax.set_xlabel(f'True {idx_name}', fontsize=10); ax.set_ylabel(f'Predicted {idx_name}', fontsize=10)
        ax.set_title(f'{idx_name} (R² = {r2_score(t, p):.3f}, Seed 42)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    plt.suptitle('Prediction vs. Ground Truth (Test Set, λ=0.1)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "fig3_pred_vs_true.png"), dpi=300, bbox_inches='tight'); plt.close()

#图 4：物理残差
if os.path.exists(res_src):
    df_res = pd.read_csv(res_src)
    fig, ax = plt.subplots(figsize=(11, 5))
    labels_r = ['Continuity', 'Thermodynamics', 'Momentum', 'Energy', 'Phase', 'Covariance']
    p6_res, nophys_res = df_res['P6Former'].values, df_res['w/o Physics'].values
    x = np.arange(len(labels_r)); width = 0.35
    ax.bar(x - width / 2, np.clip(p6_res, 1e-10, None), width, label='P⁶Former', color='#2E86AB', alpha=0.85, edgecolor='black')
    ax.bar(x + width / 2, nophys_res, width, label='w/o Physics', color='#C73E1D', alpha=0.85, edgecolor='black')
    ax.set_yscale('log'); ax.set_ylabel('Residual (log scale)', fontsize=12)
    ax.set_title('Physics Constraint Residuals', fontsize=13, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels_r, fontsize=10)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis='y', which='both')
    plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "fig4_residuals.png"), dpi=300, bbox_inches='tight'); plt.close()

#图 5 / 5b：消融实验 ----
if df_eval is not None:
    def get_per_idx(cfg):
        sub = df_eval[df_eval['Config'].str.endswith(cfg)]
        if sub.empty: return [0] * 6
        return sub[[f'{idx}_R2' for idx in INDEX_NAMES]].mean().values

    phys_per, nophys_per, noaug_per, baseline_per = get_per_idx('_phys'), get_per_idx('_nophys'), get_per_idx('_noaug'), get_per_idx('_baseline')

    fig, ax = plt.subplots(figsize=(12, 5)); x = np.arange(len(INDEX_NAMES)); width = 0.2
    configs = [('Full', phys_per, '#2E86AB'), ('w/o Physics', nophys_per, '#A23B72'), ('w/o Augmentation', noaug_per, '#F18F01'), ('Baseline', baseline_per, '#C73E1D')]
    for i, (label, values, color) in enumerate(configs):
        offset = (i - len(configs) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label, color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('R² (Test Set)'); ax.set_title('(a) Ablation Study: R² Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(INDEX_NAMES); ax.legend(loc='lower right'); ax.grid(True, alpha=0.3, axis='y'); ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "fig5_ablation_r2.png"), dpi=300, bbox_inches='tight'); plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Full\n(Physics+Aug)', 'w/o\nPhysics', 'w/o\nAugmentation', 'Baseline']
    avg_r2 = [phys_mean, nophys_mean, noaug_mean, baseline_mean]; colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    bars = ax.bar(labels, avg_r2, color=colors, alpha=0.85, edgecolor='black', linewidth=1)
    ax.set_ylabel('Average R² (Test Set)'); ax.set_title('(b) Ablation: Average R²', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim(0, 0.95)
    for bar, val in zip(bars, avg_r2):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "fig5b_ablation_avg_r2.png"), dpi=300, bbox_inches='tight'); plt.close()

#图 6：空间分布对比
if real_true_maps is not None:
    sample_idx = 0
    fig, axes = plt.subplots(3, 3, figsize=(16, 15), gridspec_kw={'hspace': 0.25})
    show_indices = [0, 2, 4]

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    X_last = X_test[:, :, :, -1, :]
    tmax_last, tmin_last, prec_last = X_last[:, :, :, 0], X_last[:, :, :, 1], X_last[:, :, :, 2]
    wet = (prec_last >= 1.0).astype(np.float32)
    pred_persist = np.zeros((X_test.shape[0], 6, 80, 140), dtype=np.float32)
    pred_persist[:, 0] = tmax_last; pred_persist[:, 1] = tmin_last
    pred_persist[:, 2] = 7.0 - wet; pred_persist[:, 3] = wet
    pred_persist[:, 4] = prec_last; pred_persist[:, 5] = np.where(wet > 0, prec_last / np.maximum(wet, 1e-6), 0.0)

    for row, i in enumerate(show_indices):
        idx_name = INDEX_NAMES[i]
        true_map, pred_map, persist_map = real_true_maps[sample_idx, i], real_pred_maps[sample_idx, i], pred_persist[sample_idx, i]
        vmin = min(true_map.min(), pred_map.min(), persist_map.min()); vmax = max(true_map.max(), pred_map.max(), persist_map.max())

        for col, (data_map, title) in enumerate([(true_map, 'True'), (pred_map, 'P⁶Former'), (persist_map, 'Persistence')]):
            ax = axes[row, col]
            im = ax.imshow(data_map, cmap='RdYlBu_r', aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_title(f'{idx_name} ({title})', fontsize=14, fontweight='bold', pad=10)
            ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f'Spatial Distribution Comparison (Test Sample #{sample_idx})', fontsize=16, fontweight='bold', y=0.92)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90]); plt.savefig(os.path.join(OUTPUT_DIR, "fig6_spatial_distribution.png"), dpi=300, bbox_inches='tight'); plt.close()

#图 7：分布漂移
try:
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")).mean(axis=(2, 3))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy")).mean(axis=(2, 3))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy")).mean(axis=(2, 3))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10)); axes = axes.flatten()
    for i, idx_name in enumerate(INDEX_NAMES):
        ax = axes[i]
        bp = ax.boxplot([y_train[:, i], y_val[:, i], y_test[:, i]], tick_labels=['Train\n(2000-2015)', 'Val\n(2016-2018)', 'Test\n(2019-2024)'], patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], ['#2E86AB', '#F18F01', '#C73E1D']):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_ylabel(f'{idx_name}'); ax.set_title(idx_name, fontsize=11, fontweight='bold'); ax.grid(True, alpha=0.3, axis='y')
    plt.suptitle('Distribution Shift Across Datasets', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "fig7_distribution_shift.png"), dpi=300, bbox_inches='tight'); plt.close()
except Exception as e:
    print(f"⚠️ 绘制图7失败: {e}")

#图 8：R95p 分层误差
if real_true_maps is not None:
    r95p_true, r95p_pred = real_true_maps[:, 4, :, :], real_pred_maps[:, 4, :, :]
    r95p_true_avg = r95p_true.mean(axis=(1, 2))
    q25, q75 = np.percentile(r95p_true_avg, 25), np.percentile(r95p_true_avg, 75)
    bottom_idx, top_idx = r95p_true_avg <= q25, r95p_true_avg >= q75

    def compute_rmse(pred, true, mask):
        err = (pred[mask] - true[mask]) ** 2
        return np.sqrt(np.mean(err))

    rmse_bottom, rmse_top = compute_rmse(r95p_pred, r95p_true, bottom_idx), compute_rmse(r95p_pred, r95p_true, top_idx)

    fig, ax = plt.subplots(figsize=(6, 5))
    categories = ['Bottom 25%\n(Weak Precipitation)', 'Top 25%\n(Extreme Precipitation)']
    rmse_values = [rmse_bottom, rmse_top]
    bars = ax.bar(categories, rmse_values, color=['#2E86AB', '#C73E1D'], width=0.5, alpha=0.85, edgecolor='black')
    for bar, val in zip(bars, rmse_values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1, f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE (mm)'); ax.set_title('R95p Prediction Error by Precipitation Intensity', fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim(0, max(rmse_values) * 1.2)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "fig8_r95p_stratify.png"), dpi=300, bbox_inches='tight'); plt.close()

print("\n🎉 全部图表生成完毕！输出目录: paper_assets/")