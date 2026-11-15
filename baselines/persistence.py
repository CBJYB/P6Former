import os
import numpy as np
from evaluate_utils import compute_metrics

DATA_DIR = r"E:\CBJdata\data2\split_7d"

print("📦 Loading test set...")
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))  # (N, 80, 140, 30, 8)
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))  # (N, 6, 80, 140)

# X shape (N, lat, lon, time, var)
# Take the last time step
X_last = X_test[:, :, :, -1, :]  # (N, 80, 140, 8)

# Variable order: tmax, tmin, prec, wind, pres, srad, lrad, shum
tmax_last = X_last[:, :, :, 0]
tmin_last = X_last[:, :, :, 1]
prec_last = X_last[:, :, :, 2]

N = X_test.shape[0]
pred = np.zeros((N, 6, 80, 140), dtype=np.float32)

# txx = tmax_last
pred[:, 0, :, :] = tmax_last
# tnn = tmin_last
pred[:, 1, :, :] = tmin_last

wet = (prec_last >= 1.0).astype(np.float32)
pred[:, 2, :, :] = 7.0 - wet
pred[:, 3, :, :] = wet
# r95p = prec_last
pred[:, 4, :, :] = prec_last
# sdii = prec_last / wet
sdii = np.where(wet > 0, prec_last / np.maximum(wet, 1e-6), 0.0)
pred[:, 5, :, :] = sdii

# Evaluation
metrics = compute_metrics(pred, y_test, "Persistence")

# Save results
np.save(os.path.join(DATA_DIR, "..", "results_persistence.npy"), metrics)
print("✅ Persistence completed")