"""
生成论文所需的所有表格和图表（Neurocomputing 终稿高级美化版 v17）。
仅修复了图 5(a) 中 0.97 标签与 TXx 重叠的问题。
运行：python total.py
"""
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

from model.config import DATA_DIR, SPATIAL_SIZE, TIME_STEPS, VARIABLES, NUM_INDICES, D_MODEL, N_HEADS, NUM_LAYERS, D_FF, \
    DROPOUT
from model.dataset import P6FormerDataset
from model.model import P6Former

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# ================= 全局绘图风格设置 =================
matplotlib.rcParams.update({
    'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 15,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12,
    'font.family': 'Arial', 'axes.unicode_minus': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.2, 'grid.linestyle': '--',
    'mathtext.fontset': 'stix'
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "paper_assets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT_DIR = os.path.join(BASE_DIR, "checkpoints_7d")
CONVLSTM_DIR = os.path.join(BASE_DIR, "data")
INDEX_NAMES = ['TXx', 'TNn', 'CDD', 'CWD', 'R95p', 'SDII']
SEEDS = [42, 123, 2024]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 丰富的高级配色方案
COLORS = {
    'Full': '#00B4D8', 'w/o Physics': '#F72585', 'w/o Aug': '#F4A261', 'Baseline': '#E63946',
}
CONSTRAINT_COLORS = ['#FF595E', '#FFCA3A', '#8AC926', '#1982C4', '#6A4C93', '#FF924C']


# 辅助函数：将 Python 的 3.1e+04 转为数学格式的 3.1 × 10^4
def format_sci(val):
    if val == 0:
        return "0.0"
    s = f"{val:.1e}"
    base, exp = s.split('e')
    exp = int(exp)
    return f"${base} \\times 10^{{{exp}}}$"


# ============================================================
# 0. 辅助工具
# ============================================================
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


# ============================================================
# 1. 加载动态数据
# ============================================================
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
    print("⚠️ 找不到 p6former_full_eval.csv")
    df_eval = None

# ============================================================
# 2. 生成表格数据
# ============================================================
res_src = os.path.join(OUTPUT_DIR, "table_physics_satisfaction.csv")
if os.path.exists(res_src):
    shutil.copy(res_src, os.path.join(OUTPUT_DIR, "table_residuals_export.csv"))

if df_eval is not None:
    ablation_r2_data = {
        'Configuration': ['Full', 'w/o Physics', 'w/o Augmentation', 'Baseline'],
        'Avg R2': [phys_mean, nophys_mean, noaug_mean, baseline_mean],
        'Std': [phys_std, nophys_std, noaug_std, baseline_std],
    }
    pd.DataFrame(ablation_r2_data).to_csv(os.path.join(OUTPUT_DIR, "table_ablation_export.csv"), index=False)

# ============================================================
# 3. 生成图表
# ============================================================
print("\n" + "=" * 70 + "\n生成图表...\n" + "=" * 70)

model_path = os.path.join(CKPT_DIR, "lam1e-1_seed42", "best_model.pth")
if os.path.exists(model_path):
    real_true_avg, real_pred_avg, real_true_maps, real_pred_maps = run_real_inference(model_path)
else:
    real_true_avg = real_pred_avg = real_true_maps = real_pred_maps = None

# ---- 图 2：研究区域 ----
if HAS_CARTOPY:
    try:
        fig = plt.figure(figsize=(10, 8));
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([73, 108, 22, 42], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='#F5F5DC');
        ax.add_feature(cfeature.OCEAN, facecolor='#E0F3F8')
        ax.add_feature(cfeature.RIVERS, linewidth=0.5, color='royalblue', alpha=0.7)
        ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='royalblue', alpha=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1, color='gray');
        ax.add_feature(cfeature.COASTLINE, linewidth=1, color='black')
        rect = Rectangle((80, 25), 25, 15, fill=False, edgecolor='red', linewidth=2, linestyle='--');
        ax.add_patch(rect)

        cities = {'Lhasa': (91.1, 29.6), 'Nagqu': (92.0, 31.4), 'Golmud': (94.9, 36.4), 'Xining': (101.8, 36.6),
                  'Shigatse': (88.8, 29.2), 'Hotan': (79.9, 37.1)}
        for name, (lon, lat) in cities.items():
            ax.plot(lon, lat, 'ro', markersize=6, transform=ccrs.PlateCarree())
            if name == 'Shigatse':
                ax.text(lon - 0.2, lat - 0.8, name, fontsize=9, fontweight='bold', transform=ccrs.PlateCarree(),
                        ha='center')
            elif name == 'Lhasa':
                ax.text(lon + 0.5, lat + 0.3, name, fontsize=9, fontweight='bold', transform=ccrs.PlateCarree())
            else:
                ax.text(lon + 0.5, lat + 0.3, name, fontsize=9, fontweight='bold', transform=ccrs.PlateCarree())

        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.4, linestyle='--',
                          color='gray')
        gl.top_labels = False;
        gl.right_labels = False
        ax.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.85), arrowprops=dict(facecolor='black', width=3, headwidth=8),
                    ha='center', va='center', fontsize=14, xycoords='axes fraction')
        plt.tight_layout();
        plt.savefig(os.path.join(OUTPUT_DIR, "fig2_study_area.png"), dpi=300, bbox_inches='tight');
        plt.close()
    except Exception as e:
        print(f"⚠️ Cartopy 渲染失败: {e}")

# ---- 图 3：散点图 (3 行 2 列竖排) ----
if real_true_avg is not None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 16));
    axes = axes.flatten()

    for i, idx_name in enumerate(INDEX_NAMES):
        ax = axes[i];
        t, p = real_true_avg[:, i], real_pred_avg[:, i]
        ax.scatter(t, p, alpha=0.6, s=25, c=COLORS['Full'], edgecolor='w', linewidth=0.5)
        mn, mx = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([mn, mx], [mn, mx], 'k--', linewidth=1.5)

        ax.set_xlabel(f'True {idx_name}', fontsize=14)
        ax.set_ylabel(f'Predicted {idx_name}', fontsize=14)

        r2_val = r2_score(t, p)
        ax.text(0.05, 0.9, f'R² = {r2_val:.3f}', transform=ax.transAxes, fontsize=14, fontweight='bold')

    plt.tight_layout();
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_pred_vs_true.png"), dpi=300, bbox_inches='tight');
    plt.close()

# ---- 图 4：物理残差 ----
if os.path.exists(res_src):
    df_res = pd.read_csv(res_src)
    labels_r = ['Continuity', 'Thermodynamics', 'Momentum', 'Energy', 'Phase', 'Covariance']
    p6_res, nophys_res = df_res['P6Former'].values, df_res['w/o Physics'].values
    x = np.arange(len(labels_r))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    bars1 = ax.bar(x - width / 2, np.clip(p6_res, 1e-10, None), width, label='P⁶Former',
                   color=CONSTRAINT_COLORS, alpha=0.9, edgecolor='black', linewidth=1, zorder=3)
    bars2 = ax.bar(x + width / 2, nophys_res, width, label='w/o Physics',
                   color=CONSTRAINT_COLORS, alpha=0.5, edgecolor='black', linewidth=1, hatch='//', zorder=3)

    ax.set_yscale('log')

    for i in range(len(labels_r)):
        val_p6 = np.clip(p6_res[i], 1e-10, None)
        ax.text(x[i] - width / 2 - 0.02, val_p6 * 1.5, format_sci(p6_res[i]), ha='right', va='bottom', fontsize=10,
                fontweight='bold')

        val_no = nophys_res[i]
        ax.text(x[i] + width / 2 + 0.02, val_no * 1.5, format_sci(nophys_res[i]), ha='left', va='bottom', fontsize=10,
                fontweight='bold', color='#444444')

    ax.set_ylabel('Residual (log scale)', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_r, fontsize=13, fontweight='bold')
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', label='P⁶Former'),
        Patch(facecolor='lightgray', edgecolor='black', hatch='//', label='w/o Physics')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12)
    ax.set_ylim(1e-11, 1e6)

    plt.tight_layout();
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_residuals.png"), dpi=300, bbox_inches='tight');
    plt.close()

    # ---- 图 5(a)：消融实验雷达图 (补全完整的同心圆虚线网格) ----
    if df_eval is not None:
        def get_per_idx(cfg):
            sub = df_eval[df_eval['Config'].str.endswith(cfg)]
            if sub.empty: return [0] * 6
            return sub[[f'{idx}_R2' for idx in INDEX_NAMES]].mean().values


        phys_per, nophys_per, noaug_per, baseline_per = get_per_idx('_phys'), get_per_idx('_nophys'), get_per_idx(
            '_noaug'), get_per_idx('_baseline')

        labels_radar = np.array(INDEX_NAMES)
        num_vars = len(labels_radar)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))
        ax.set_facecolor('#FFFFFF')

        # 🔧 核心修复 1：统一成极其清晰的浅灰虚线，线宽调细
        ax.xaxis.grid(color='#BDBDBD', linestyle='--', linewidth=1.2, alpha=1.0, zorder=1)
        ax.yaxis.grid(color='#BDBDBD', linestyle='--', linewidth=1.2, alpha=1.0, zorder=1)


        def close_data(data):
            return data.tolist() + [data[0]]


        # 数据线画在网格线之上
        ax.plot(angles, close_data(baseline_per), linewidth=3, linestyle='-.', label='Baseline',
                color=COLORS['Baseline'], marker='D', markersize=8, alpha=0.9, zorder=3)
        ax.plot(angles, close_data(noaug_per), linewidth=3, linestyle='--', label='w/o Augmentation',
                color=COLORS['w/o Aug'], marker='^', markersize=8, alpha=0.9, zorder=4)
        ax.plot(angles, close_data(nophys_per), linewidth=3, linestyle=':', label='w/o Physics',
                color=COLORS['w/o Physics'], marker='s', markersize=8, alpha=0.9, zorder=5)
        ax.plot(angles, close_data(phys_per), linewidth=4, linestyle='-', label='Full', color=COLORS['Full'],
                marker='o', markersize=10, zorder=6)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_radar, fontsize=22, fontweight='bold')
        ax.tick_params(axis='x', pad=25)

        # 🔧 核心修复 2：必须设置 yticks 才能画出完整的同心圆网格
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

        # 把系统自带的刻度文字隐藏，只留网格线
        ax.set_yticklabels([])

        # 手动把刻度文字写到 45 度角方向
        manual_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
        text_angle = np.pi / 4
        for tick_val in manual_ticks:
            ax.text(text_angle, tick_val, f"{tick_val:.1f}", ha='left', va='center',
                    fontsize=16, fontweight='bold', color='#333333',
                    bbox=dict(facecolor='white', alpha=1.0, edgecolor='#BDBDBD', pad=3),
                    zorder=10)

            # 数据标签
        for i, val in enumerate(phys_per):
            angle_rad = angles[i]
            y_pos = val - 0.1 if val > 0.9 else val + 0.06
            ax.text(angle_rad, y_pos, f'{val:.2f}', ha='center', va='center',
                    fontsize=18, fontweight='bold', color=COLORS['Full'],
                    bbox=dict(facecolor='white', alpha=1.0, edgecolor=COLORS['Full'], linewidth=1.5, pad=3),
                    zorder=10)

        ax.set_title('(a) Ablation Study: R² Radar Chart', fontsize=24, fontweight='bold', pad=40)
        plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=16)
        plt.tight_layout();
        plt.savefig(os.path.join(OUTPUT_DIR, "fig5a_ablation_radar.png"), dpi=300, bbox_inches='tight');
        plt.close()

    # ---- 图 5(b)：极坐标玫瑰图 ----
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, polar=True)

    labels_bar = ['Baseline', 'w/o Aug', 'w/o Physics', 'Full']
    means = [baseline_mean, noaug_mean, nophys_mean, phys_mean]
    colors_bar = [COLORS['Baseline'], COLORS['w/o Aug'], COLORS['w/o Physics'], COLORS['Full']]

    N = len(labels_bar)
    offset = np.pi / 4
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False) + offset
    width = np.pi / 4 * 0.8

    bars = ax.bar(theta, means, width=width, bottom=0.0, color=colors_bar, alpha=0.85, edgecolor='black', linewidth=2)

    ax.set_xticks(theta)
    ax.set_xticklabels(labels_bar, fontsize=20, fontweight='bold')
    ax.tick_params(axis='x', pad=25)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="#333333", size=16, fontweight='bold')
    ax.set_rlabel_position(22.5)

    ax.xaxis.grid(color='black', linestyle='--', linewidth=1.5)
    ax.yaxis.grid(color='black', linestyle='--', linewidth=1.5)

    ax.set_facecolor('#F8F9FA')

    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='none', pad=3))

    for angle, mean, bar in zip(theta, means, bars):
        y_pos = mean - 0.07
        ax.text(angle, y_pos, f'{mean:.4f}', ha='center', va='center',
                fontsize=18, fontweight='bold',
                bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', pad=2))

    ax.set_title('(b) Ablation Study: Average R²', fontsize=22, fontweight='bold', pad=30)
    plt.tight_layout();
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5b_ablation_avg_r2.png"), dpi=300, bbox_inches='tight');
    plt.close()

# ---- 图 6：空间分布对比 ----
if real_true_maps is not None:
    sample_idx = 0
    fig, axes = plt.subplots(3, 3, figsize=(15, 14), gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
    show_indices = [0, 2, 4]

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    X_last = X_test[:, :, :, -1, :]
    tmax_last, tmin_last, prec_last = X_last[:, :, :, 0], X_last[:, :, :, 1], X_last[:, :, :, 2]
    wet = (prec_last >= 1.0).astype(np.float32)
    pred_persist = np.zeros((X_test.shape[0], 6, 80, 140), dtype=np.float32)
    pred_persist[:, 0] = tmax_last;
    pred_persist[:, 1] = tmin_last
    pred_persist[:, 2] = 7.0 - wet;
    pred_persist[:, 3] = wet
    pred_persist[:, 4] = prec_last;
    pred_persist[:, 5] = np.where(wet > 0, prec_last / np.maximum(wet, 1e-6), 0.0)

    for row, i in enumerate(show_indices):
        idx_name = INDEX_NAMES[i]
        true_map, pred_map, persist_map = real_true_maps[sample_idx, i], real_pred_maps[sample_idx, i], pred_persist[
            sample_idx, i]
        vmin = min(true_map.min(), pred_map.min(), persist_map.min());
        vmax = max(true_map.max(), pred_map.max(), persist_map.max())

        for col, (data_map, title) in enumerate(
                [(true_map, 'True'), (pred_map, 'P⁶Former'), (persist_map, 'Persistence')]):
            ax = axes[row, col]
            im = ax.imshow(data_map, cmap='RdYlBu_r', aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_ylabel(f'{idx_name}\n({title})', fontsize=14, fontweight='bold')
            ax.set_xticks([]);
            ax.set_yticks([])
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=11)

    plt.tight_layout();
    plt.savefig(os.path.join(OUTPUT_DIR, "fig6_spatial_distribution.png"), dpi=300, bbox_inches='tight');
    plt.close()

# ---- 图 7：分布漂移 (3 行 2 列竖排) ----
try:
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")).mean(axis=(2, 3))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy")).mean(axis=(2, 3))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy")).mean(axis=(2, 3))

    fig, axes = plt.subplots(3, 2, figsize=(12, 16));
    axes = axes.flatten()

    for i, idx_name in enumerate(INDEX_NAMES):
        ax = axes[i]
        bp = ax.boxplot([y_train[:, i], y_val[:, i], y_test[:, i]],
                        tick_labels=['Train\n(2000-2015)', 'Val\n(2016-2018)', 'Test\n(2019-2024)'], patch_artist=True,
                        widths=0.5)
        for patch, color in zip(bp['boxes'], [COLORS['Full'], COLORS['w/o Aug'], COLORS['Baseline']]):
            patch.set_facecolor(color);
            patch.set_alpha(0.7)

        ax.set_ylabel(f'{idx_name}', fontsize=14)
        ax.tick_params(axis='x', labelsize=12)

    plt.tight_layout();
    plt.savefig(os.path.join(OUTPUT_DIR, "fig7_distribution_shift.png"), dpi=300, bbox_inches='tight');
    plt.close()
except Exception as e:
    print(f"⚠️ 绘制图7失败: {e}")

# ---- 图 8：R95p 分层误差 ----
if real_true_maps is not None:
    r95p_true, r95p_pred = real_true_maps[:, 4, :, :], real_pred_maps[:, 4, :, :]
    r95p_true_avg = r95p_true.mean(axis=(1, 2))
    q25, q75 = np.percentile(r95p_true_avg, 25), np.percentile(r95p_true_avg, 75)
    bottom_idx, top_idx = r95p_true_avg <= q25, r95p_true_avg >= q75

    err_abs = (r95p_pred - r95p_true) ** 2
    rmse_per_sample = np.sqrt(err_abs.mean(axis=(1, 2)))

    data_bottom = rmse_per_sample[bottom_idx]
    data_top = rmse_per_sample[top_idx]

    fig, ax = plt.subplots(figsize=(6, 8))
    parts = ax.violinplot([data_bottom, data_top], showmeans=False, showmedians=True, showextrema=False, widths=0.6)

    for pc, color in zip(parts['bodies'], [COLORS['Full'], COLORS['Baseline']]):
        pc.set_facecolor(color);
        pc.set_edgecolor('black');
        pc.set_alpha(0.4)

    bp = ax.boxplot([data_bottom, data_top], tick_labels=['Bottom 25%\n(Weak)', 'Top 25%\n(Extreme)'],
                    patch_artist=True, widths=0.15, showfliers=False)
    for patch, color in zip(bp['boxes'], [COLORS['Full'], COLORS['Baseline']]):
        patch.set_facecolor(color);
        patch.set_alpha(0.8)

    x_bottom = np.random.normal(1, 0.05, size=len(data_bottom))
    x_top = np.random.normal(2, 0.05, size=len(data_top))
    ax.scatter(x_bottom, data_bottom, alpha=0.3, s=15, color=COLORS['Full'], edgecolor='k', linewidth=0.5)
    ax.scatter(x_top, data_top, alpha=0.3, s=15, color=COLORS['Baseline'], edgecolor='k', linewidth=0.5)

    ax.set_ylabel('RMSE per Sample (mm)', fontsize=15)
    ax.set_ylim(0, max(data_top) * 1.1)
    plt.tight_layout();
    plt.savefig(os.path.join(OUTPUT_DIR, "fig8_r95p_stratify.png"), dpi=300, bbox_inches='tight');
    plt.close()

print("\n🎉 全部图表生成完毕！输出目录: paper_assets/")