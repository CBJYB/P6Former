import os
import numpy as np
from evaluate_utils import compute_metrics

DATA_DIR = r"E:\CBJdata\data2\split_7d"

print("📦 加载测试集...")
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))  # (N, 80, 140, 30, 8)
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))  # (N, 6, 80, 140)

# X 形状 (N, lat, lon, time, var)
# 取最后一个时间步
X_last = X_test[:, :, :, -1, :]  # (N, 80, 140, 8)

# 变量顺序: tmax, tmin, prec, wind, pres, srad, lrad, shum
tmax_last = X_last[:, :, :, 0]
tmin_last = X_last[:, :, :, 1]
prec_last = X_last[:, :, :, 2]

N = X_test.shape[0]
pred = np.zeros((N, 6, 80, 140), dtype=np.float32)

# txx = tmax_last
pred[:, 0, :, :] = tmax_last
# tnn = tmin_last
pred[:, 1, :, :] = tmin_last
# cdd = 7 - (prec_last >= 1).sum(dim=?) 这里 prec_last 是单时刻，简化为 7 - wet
wet = (prec_last >= 1.0).astype(np.float32)
pred[:, 2, :, :] = 7.0 - wet  # cdd 近似
pred[:, 3, :, :] = wet        # cwd 近似
# r95p = prec_last (简化)
pred[:, 4, :, :] = prec_last
# sdii = prec_last / wet (避免除零)
sdii = np.where(wet > 0, prec_last / np.maximum(wet, 1e-6), 0.0)
pred[:, 5, :, :] = sdii

# 评估
metrics = compute_metrics(pred, y_test, "Persistence")

# 保存结果
np.save(os.path.join(DATA_DIR, "..", "results_persistence.npy"), metrics)
print("✅ Persistence 完成")