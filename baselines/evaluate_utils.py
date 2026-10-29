import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

INDICES_NAMES = ['txx', 'tnn', 'cdd', 'cwd', 'r95p', 'sdii']

def compute_metrics(pred, true, name=""):
    """
    pred, true: (N, 6, H, W)
    返回每个指数的 RMSE, MAE, R²
    """
    # 空间平均
    pred_avg = pred.mean(axis=(2, 3))  # (N, 6)
    true_avg = true.mean(axis=(2, 3))  # (N, 6)

    rmse_list, mae_list, r2_list = [], [], []
    for i in range(len(INDICES_NAMES)):
        rmse = np.sqrt(mean_squared_error(true_avg[:, i], pred_avg[:, i]))
        mae = mean_absolute_error(true_avg[:, i], pred_avg[:, i])
        r2 = r2_score(true_avg[:, i], pred_avg[:, i])
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)

    print(f"\n📊 {name} Metrics")
    print(f"{'Index':<6} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print("-" * 40)
    for i, idx in enumerate(INDICES_NAMES):
        print(f"{idx:<6} {rmse_list[i]:<10.4f} {mae_list[i]:<10.4f} {r2_list[i]:<10.4f}")
    print("-" * 40)
    print(f"{'Mean':<6} {np.mean(rmse_list):<10.4f} {np.mean(mae_list):<10.4f} {np.mean(r2_list):<10.4f}")

    return {
        'rmse': rmse_list,
        'mae': mae_list,
        'r2': r2_list,
        'rmse_avg': np.mean(rmse_list),
        'mae_avg': np.mean(mae_list),
        'r2_avg': np.mean(r2_list)
    }