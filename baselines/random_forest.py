import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm
from evaluate_utils import compute_metrics

DATA_DIR = r"E:\CBJdata\data2\split_7d"

print("📦 加载数据集...")
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

# X: (N, 80, 140, 30, 8)
# 提取特征：对每个样本，每个变量计算时间均值、时间标准差、空间均值、空间标准差
def extract_features(X):
    N = X.shape[0]
    feats = []
    for i in tqdm(range(N), desc="Extracting features"):
        x = X[i]  # (80, 140, 30, 8)
        # 时间均值 (80,140,8) -> 空间均值 (8,)
        t_mean = x.mean(axis=2)  # (80,140,8)
        # 时间标准差
        t_std = x.std(axis=2)    # (80,140,8)
        # 空间均值
        s_mean = t_mean.mean(axis=(0,1))  # (8,)
        # 空间标准差
        s_std = t_mean.std(axis=(0,1))    # (8,)
        # 全局最大/最小
        t_max = x.max(axis=2).mean(axis=(0,1))  # (8,)
        t_min = x.min(axis=2).mean(axis=(0,1))  # (8,)
        feat = np.concatenate([s_mean, s_std, t_max, t_min])  # (32,)
        feats.append(feat)
    return np.array(feats)

X_train_feat = extract_features(X_train)
X_test_feat = extract_features(X_test)

# y: (N, 6, 80, 140) -> 空间平均 (N, 6)
y_train_avg = y_train.mean(axis=(2,3))
y_test_avg = y_test.mean(axis=(2,3))

print("🌲 训练 Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
model = MultiOutputRegressor(rf)
model.fit(X_train_feat, y_train_avg)

print("🔮 预测...")
y_pred_avg = model.predict(X_test_feat)  # (N, 6)

# 将预测扩展到空间场（直接复制）
N = y_test.shape[0]
pred = np.zeros((N, 6, 80, 140), dtype=np.float32)
for i in range(N):
    for j in range(6):
        pred[i, j, :, :] = y_pred_avg[i, j]

# 评估
metrics = compute_metrics(pred, y_test, "Random Forest")

# 保存结果
np.save(os.path.join(DATA_DIR, "..", "results_rf.npy"), metrics)
print("✅ Random Forest 完成")