import os
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .config import *
from .dataset import P6FormerDataset
from .model import P6Former
from .physics_loss import PhysicsLoss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


print("=" * 60)
print(f"🔬 Experiment: {EXPERIMENT_NAME}")
print(f"   Seed: {SEED}")
print(f"   Dummy Mode: {USE_DUMMY}")
print(f"   Save Path: {MODEL_SAVE_DIR}")
print("=" * 60)


def compute_normalization(dataset, n_samples=100):
    print("📊 Computing normalization parameters...")
    original_augment = dataset.augment
    dataset.augment = False
    all_X = []
    n = min(len(dataset), n_samples)
    for i in range(n):
        X, _ = dataset[i]
        all_X.append(X.numpy())
    dataset.augment = original_augment

    all_X = np.stack(all_X, axis=0)
    var_mean = all_X.mean(axis=(0, 1, 3, 4))
    var_std = all_X.std(axis=(0, 1, 3, 4))

    print(f"   ✅ Mean: {var_mean.round(2)}")
    print(f"   ✅ Std:  {var_std.round(2)}")
    return var_mean, var_std


def normalize(x, var_mean, var_std):
    m = torch.tensor(var_mean, dtype=torch.float32, device=x.device).view(1, 1, -1, 1, 1)
    s = torch.tensor(var_std, dtype=torch.float32, device=x.device).view(1, 1, -1, 1, 1)
    return (x - m) / (s + 1e-8)


def get_warmup_factor(epoch, warmup_epochs, ramp_epochs=20):
    if epoch < warmup_epochs:
        return 0.0
    return min(1.0, (epoch - warmup_epochs) / float(ramp_epochs))


def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("📦 Loading datasets...")
    train_dataset = P6FormerDataset(
        x_file=os.path.join(DATA_DIR, "X_train.npy"),
        y_file=os.path.join(DATA_DIR, "y_train.npy"),
        indices_names=['txx', 'tnn', 'cdd', 'cwd', 'r95p', 'sdii'],
        target_spatial=SPATIAL_SIZE, augment=USE_AUGMENT
    )
    val_dataset = P6FormerDataset(
        x_file=os.path.join(DATA_DIR, "X_val.npy"),
        y_file=os.path.join(DATA_DIR, "y_val.npy"),
        indices_names=['txx', 'tnn', 'cdd', 'cwd', 'r95p', 'sdii'],
        target_spatial=SPATIAL_SIZE, augment=False
    )

    var_mean, var_std = compute_normalization(train_dataset)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=PIN_MEMORY)

    model = P6Former(
        time_steps=TIME_STEPS, in_vars=VARIABLES, num_indices=NUM_INDICES,
        d_model=D_MODEL, n_heads=N_HEADS, num_layers=NUM_LAYERS,
        d_ff=D_FF, dropout=DROPOUT, spatial_size=SPATIAL_SIZE
    ).to(device)

    physics_loss_fn = PhysicsLoss(enabled=ENABLED_CONSTRAINTS, var_mean=var_mean, var_std=var_std).to(device)
    log_vars = nn.Parameter(torch.zeros(6, device=device))

    optimizer = AdamW([
        {'params': model.parameters(), 'lr': LEARNING_RATE},
        {'params': [log_vars], 'lr': LEARNING_RATE * 0.1},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print("🚀 Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, total_task_loss, total_phys_loss = 0.0, 0.0, 0.0
        warmup_factor = get_warmup_factor(epoch, WARMUP_EPOCHS, ramp_epochs=20)

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]"):
            X = normalize(X, var_mean, var_std).to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred_y, aux_out = model(X)
            task_loss = nn.MSELoss()(pred_y, y)

            if USE_PHYSICS and warmup_factor > 0:
                phys_dict = physics_loss_fn(aux_out)
                phys_loss = torch.tensor(0.0, device=device)
                for i, (key, val) in enumerate(phys_dict.items()):
                    if val.item() > 1e-12:
                        phys_loss = phys_loss + torch.exp(-log_vars[i]) * val + log_vars[i]
                phys_loss = phys_loss * PHYSICS_SCALE * warmup_factor
            else:
                phys_loss = torch.tensor(0.0, device=device)

            loss = task_loss + phys_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [log_vars], GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_phys_loss += phys_loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X = normalize(X, var_mean, var_std).to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred_y, _ = model(X)
                total_val_loss += nn.MSELoss()(pred_y, y).item()
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch + 1:3d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_model.pth"))

            summary = {
                'experiment': EXPERIMENT_NAME,
                'seed': SEED,
                'use_physics': USE_PHYSICS,
                'use_augment': USE_AUGMENT,
                'var_mean': var_mean,
                'var_std': var_std,
            }
            np.save(os.path.join(MODEL_SAVE_DIR, "summary.npy"), summary)

        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"⏹️ Early stopping at epoch {epoch + 1}")
                break

    print("🎉 Training completed!")


if __name__ == "__main__":
    main()