import os
import numpy as np
from evaluate_utils import compute_metrics

DATA_DIR = r"E:\CBJdata\data2\split_7d"

print("📦 加载训练集和测试集...")
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))  # (N, 6, 80, 140)
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))    # (N, 6, 80, 140)

# 计算训练集均值
y_mean = y_train.mean(axis=0)  # (6, 80, 140)

# 对每个测试样本，预测值都是 y_mean
N = y_test.shape[0]
pred = np.tile(y_mean[np.newaxis, ...], (N, 1, 1, 1))

# 评估
metrics = compute_metrics(pred, y_test, "Climatology")

# 保存结果
np.save(os.path.join(DATA_DIR, "..", "results_climatology.npy"), metrics)
print("✅ Climatology 完成")