import torch
from torch.utils.data import Dataset
import numpy as np

class P6FormerDataset(Dataset):
    def __init__(self, x_file, y_file, indices_names, target_spatial=(80, 140), time_steps=30, variables=8, augment=False):
        print(f"📦 加载 {x_file}...")
        self.X = np.load(x_file, mmap_mode='r')
        self.y = np.load(y_file)
        if self.X.shape[3] > time_steps:
            self.X = self.X[:, :, :, -time_steps:, :]
        self.time_steps = time_steps; self.variables = variables; self.indices_names = indices_names
        self.augment = augment; self.target_spatial = target_spatial
        print(f"   ✅ 数据集加载完成: X {self.X.shape}, y {self.y.shape}")

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        X = np.nan_to_num(self.X[idx], nan=0.0).astype(np.float32)
        y = np.nan_to_num(self.y[idx], nan=0.0).astype(np.float32)
        if self.augment:
            if np.random.rand() > 0.5:
                X = np.flip(X, axis=1); y = np.flip(y, axis=2)
            if np.random.rand() > 0.5:
                X = np.flip(X, axis=0); y = np.flip(y, axis=1)
            X = np.ascontiguousarray(X); y = np.ascontiguousarray(y)
        return torch.from_numpy(X).float().permute(2, 3, 0, 1), torch.from_numpy(y).float()