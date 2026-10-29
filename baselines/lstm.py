import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from evaluate_utils import compute_metrics

# 配置
DATA_DIR = r"E:\CBJdata\data2\split_7d"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
HIDDEN_DIM = 128
NUM_LAYERS = 2
OUTPUT_DIM = 6  # 6 个指数

print(f"🔧 使用设备: {DEVICE}")

# 加载数据
print("📦 加载数据集...")
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

# 特征提取：对每个时间步，计算每个变量的空间均值 -> (N, T, 8)
def extract_seq_features(X):
    # X: (N, 80, 140, 30, 8)
    # 空间平均 -> (N, 30, 8)
    X_spatial_mean = X.mean(axis=(1, 2))  # (N, 30, 8)
    return X_spatial_mean

X_train_seq = extract_seq_features(X_train)
X_val_seq = extract_seq_features(X_val)
X_test_seq = extract_seq_features(X_test)

# y: (N, 6, 80, 140) -> 空间平均 (N, 6)
y_train_avg = y_train.mean(axis=(2, 3))
y_val_avg = y_val.mean(axis=(2, 3))
y_test_avg = y_test.mean(axis=(2, 3))

# 转 tensor
X_train_t = torch.from_numpy(X_train_seq).float()
y_train_t = torch.from_numpy(y_train_avg).float()
X_val_t = torch.from_numpy(X_val_seq).float()
y_val_t = torch.from_numpy(y_val_avg).float()
X_test_t = torch.from_numpy(X_test_seq).float()
y_test_t = torch.from_numpy(y_test_avg).float()

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, num_layers=2, output_dim=6):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, T, input_dim)
        out, _ = self.lstm(x)  # (B, T, hidden_dim)
        out = out[:, -1, :]    # 取最后一个时间步
        return self.fc(out)

model = LSTMModel(input_dim=8, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=OUTPUT_DIM).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 训练
print("🚀 开始训练 LSTM...")
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            pred = model(X_batch)
            val_loss += criterion(pred, y_batch).item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(DATA_DIR, "..", "best_lstm.pth"))

print("✅ LSTM 训练完成")

# 测试
print("🔮 测试集预测...")
model.load_state_dict(torch.load(os.path.join(DATA_DIR, "..", "best_lstm.pth")))
model.eval()
preds = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(DEVICE)
        pred = model(X_batch).cpu().numpy()
        preds.append(pred)
y_pred_avg = np.concatenate(preds, axis=0)  # (N, 6)

# 扩展为空间场
N = y_test.shape[0]
pred = np.zeros((N, 6, 80, 140), dtype=np.float32)
for i in range(N):
    for j in range(6):
        pred[i, j, :, :] = y_pred_avg[i, j]

# 评估
metrics = compute_metrics(pred, y_test, "LSTM")

# 保存结果
np.save(os.path.join(DATA_DIR, "..", "results_lstm.npy"), metrics)
print("✅ LSTM 完成")