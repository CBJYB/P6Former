import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.physics_loss import PhysicsLoss
from evaluate_utils import compute_metrics

SEED = int(os.environ.get('SEED', 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"🎲 SEED = {SEED}")

DATA_DIR = r"E:\CBJdata\data2\split_7d"
SAVE_PATH = rf"E:\CBJdata\data2\best_phy_convlstm_seed{SEED}.pth"
RESULT_PATH = rf"E:\CBJdata\data2\results_phy_convlstm_seed{SEED}.npy"
PRED_PATH = rf"E:\CBJdata\data2\predictions_phy_convlstm_seed{SEED}.npy"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4
EPOCHS = 200       # 🔧
LR = 5e-4
GRAD_CLIP = 1.0
EARLY_STOP = 50    # 🔧
WARMUP_EPOCHS = 50
PHYSICS_SCALE = 0.1 # 🔧 统一为0.1

print(f"🔧 Device: {DEVICE}")

class ConvLSTMDataset(Dataset):
    def __init__(self, x_file, y_file, var_mean=None, var_std=None):
        self.X = np.load(x_file, mmap_mode='r')
        self.y = np.load(y_file)
        self.var_mean = var_mean
        self.var_std = var_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = np.transpose(self.X[idx], (2, 3, 0, 1)).astype(np.float32)
        if self.var_mean is not None and self.var_std is not None:
            m = self.var_mean.reshape(1, -1, 1, 1)
            s = self.var_std.reshape(1, -1, 1, 1)
            x = (x - m) / (s + 1e-8)
        y = self.y[idx].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

print("📦 加载数据集...")
x_train_path = os.path.join(DATA_DIR, "X_train.npy")
y_train_path = os.path.join(DATA_DIR, "y_train.npy")
x_val_path = os.path.join(DATA_DIR, "X_val.npy")
y_val_path = os.path.join(DATA_DIR, "y_val.npy")
x_test_path = os.path.join(DATA_DIR, "X_test.npy")
y_test_path = os.path.join(DATA_DIR, "y_test.npy")

print("📊 Computing per-variable normalization parameters...")
X_train_mmap = np.load(x_train_path, mmap_mode='r')
step = max(1, len(X_train_mmap) // 100)
sample_vals = [np.transpose(X_train_mmap[i], (2, 3, 0, 1)) for i in range(0, len(X_train_mmap), step)]
sample_vals = np.concatenate(sample_vals, axis=0)
var_mean = sample_vals.mean(axis=(0, 2, 3))
var_std = sample_vals.std(axis=(0, 2, 3))
del sample_vals, X_train_mmap

train_dataset = ConvLSTMDataset(x_train_path, y_train_path, var_mean, var_std)
val_dataset = ConvLSTMDataset(x_val_path, y_val_path, var_mean, var_std)
test_dataset = ConvLSTMDataset(x_test_path, y_test_path, var_mean, var_std)

g = torch.Generator(); g.manual_seed(SEED)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=g)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2, output_dim=6):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(ConvLSTMCell(in_dim, hidden_dim))
        self.output_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, output_dim, kernel_size=1)
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        h = [torch.zeros(B, self.hidden_dim, H, W, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_dim, H, W, device=x.device) for _ in range(self.num_layers)]
        for t in range(T):
            inp = x[:, t]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](inp, h[layer], c[layer])
                inp = h[layer]
        main_out = self.output_head(h[-1])
        aux_out = self.aux_head(h[-1])
        return main_out, aux_out

model = ConvLSTM(input_dim=8, hidden_dim=64, num_layers=2, output_dim=6).to(DEVICE)
print(f"📊 PhyConvLSTM parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

# 🔧 传入逐变量参数，执行真实物理计算
physics_loss_fn = PhysicsLoss(var_mean=var_mean, var_std=var_std).to(DEVICE)
log_vars = nn.Parameter(torch.zeros(6, device=DEVICE))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(model.parameters()) + [log_vars], lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print("\n🚀 Training PhyConvLSTM...")
best_val_loss = float('inf')
patience = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        pred, aux = model(X_batch)
        task_loss = criterion(pred, y_batch)

        if epoch >= WARMUP_EPOCHS:
            phys_dict = physics_loss_fn(aux)
            phys_loss = 0
            for i, (k, v) in enumerate(phys_dict.items()):
                phys_loss += torch.exp(-log_vars[i]) * v + log_vars[i]
            phys_loss = phys_loss * PHYSICS_SCALE
        else:
            phys_loss = torch.tensor(0.0, device=DEVICE)

        loss = task_loss + phys_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [log_vars], GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()

    avg_train = total_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            pred, _ = model(X_batch)
            val_loss += criterion(pred, y_batch).item()
    avg_val = val_loss / len(val_loader)
    scheduler.step()

    print(f"Epoch {epoch+1:3d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        patience = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"   ✅ Best saved (Val: {avg_val:.4f})")
    else:
        patience += 1
        if patience >= EARLY_STOP:
            print("⏹️ Early stopping")
            break

print("\n🔮 Testing PhyConvLSTM...")
model.load_state_dict(torch.load(SAVE_PATH))
model.eval()
preds, targets = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        pred, _ = model(X_batch)
        preds.append(pred.cpu().numpy())
        targets.append(y_batch.numpy())

pred = np.concatenate(preds, axis=0)
true = np.concatenate(targets, axis=0)

np.save(PRED_PATH, pred) # 保存预测值
metrics = compute_metrics(pred, true, "PhyConvLSTM")
np.save(RESULT_PATH, metrics)
print(f"✅ PhyConvLSTM seed={SEED} 完成。")