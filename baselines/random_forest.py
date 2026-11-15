import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm
from evaluate_utils import compute_metrics

DATA_DIR = r"E:\CBJdata\data2\split_7d"

print("📦 Loading dataset...")
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

# X: (N, 80, 140, 30, 8)
# Feature extraction: for each sample and each variable, compute temporal mean, temporal standard deviation, spatial mean, and spatial standard deviation
def extract_features(X):
    N = X.shape[0]
    feats = []
    for i in tqdm(range(N), desc="Extracting features"):
        x = X[i]  # (80, 140, 30, 8)
        # Temporal mean (80,140,8) -> spatial mean (8,)
        t_mean = x.mean(axis=2)  # (80,140,8)
        # Temporal standard deviation
        t_std = x.std(axis=2)    # (80,140,8)
        # Spatial mean
        s_mean = t_mean.mean(axis=(0,1))  # (8,)
        # Spatial standard deviation
        s_std = t_mean.std(axis=(0,1))    # (8,)
        # Global maximum/minimum
        t_max = x.max(axis=2).mean(axis=(0,1))  # (8,)
        t_min = x.min(axis=2).mean(axis=(0,1))  # (8,)
        feat = np.concatenate([s_mean, s_std, t_max, t_min])  # (32,)
        feats.append(feat)
    return np.array(feats)

X_train_feat = extract_features(X_train)
X_test_feat = extract_features(X_test)

# y: (N, 6, 80, 140) -> spatial average (N, 6)
y_train_avg = y_train.mean(axis=(2,3))
y_test_avg = y_test.mean(axis=(2,3))

print("🌲 Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
model = MultiOutputRegressor(rf)
model.fit(X_train_feat, y_train_avg)

print("🔮 Predicting...")
y_pred_avg = model.predict(X_test_feat)  # (N, 6)

# Expand predictions to spatial fields
N = y_test.shape[0]
pred = np.zeros((N, 6, 80, 140), dtype=np.float32)
for i in range(N):
    for j in range(6):
        pred[i, j, :, :] = y_pred_avg[i, j]

# Evaluation
metrics = compute_metrics(pred, y_test, "Random Forest")

# Save results
np.save(os.path.join(DATA_DIR, "..", "results_rf.npy"), metrics)
print("✅ Random Forest completed")