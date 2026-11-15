import os
import numpy as np
from evaluate_utils import compute_metrics

DATA_DIR = r"E:\CBJdata\data2\split_7d"

print("📦 Loading training and test sets...")
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))  # (N, 6, 80, 140)
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))    # (N, 6, 80, 140)

# Compute training-set mean
y_mean = y_train.mean(axis=0)  # (6, 80, 140)

# For each test sample, the prediction is y_mean
N = y_test.shape[0]
pred = np.tile(y_mean[np.newaxis, ...], (N, 1, 1, 1))

# Evaluation
metrics = compute_metrics(pred, y_test, "Climatology")

# Save results
np.save(os.path.join(DATA_DIR, "..", "results_climatology.npy"), metrics)
print("✅ Climatology completed")