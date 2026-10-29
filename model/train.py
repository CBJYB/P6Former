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
print(f"🔬 实验名称: {EXPERIMENT_NAME}")
print(f"   随机种子: {SEED}")
print(f"   物理约束: {'✅ 启用' if USE_PHYSICS else '❌ 禁用'}")
_enabled_str = "全部 6 个" if ENABLED_CONSTRAINTS is None else f"{sorted(ENABLED_CONSTRAINTS)}"
print(f"   启用约束: {_enabled_str}")
print(f"   数据增强: {'✅ 启用' if USE_AUGMENT else '❌ 禁用'}")
print(f"   物理缩放: {PHYSICS_SCALE}")
print(f"   热启动轮数: {WARMUP_EPOCHS}")
print(f"   保存路径: {MODEL_SAVE_DIR}")
print("=" * 60)


def compute_normalization(dataset, n_samples=100):
    print("📊 计算逐变量归一化参数...")
    original_augment = dataset.augment
    dataset.augment = False
    all_X = []
    n = min(len(dataset), n_samples)
    for i in range(n):
        X, _ = dataset[i]
        all_X.append(X.numpy())
    dataset.augment = original_augment

    all_X = np.stack(all_X, axis=0)
    var_mean = all_X.mean(axis=(0, 1, 3, 4))  # shape: (8,)
    var_std = all_X.std(axis=(0, 1, 3, 4))  # shape: (8,)

    print(f"   ✅ 逐变量 Mean: {var_mean.round(2)}")
    print(f"   ✅ 逐变量 Std:  {var_std.round(2)}")
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
    print(f"🔧 使用设备: {device}")
    data_dir = DATA_DIR

    print("📦 加载数据集...")
    train_dataset = P6FormerDataset(
        x_file=os.path.join(data_dir, "X_train.npy"),
        y_file=os.path.join(data_dir, "y_train.npy"),
        indices_names=['txx', 'tnn', 'cdd', 'cwd', 'r95p', 'sdii'],
        target_spatial=SPATIAL_SIZE, augment=USE_AUGMENT
    )
    val_dataset = P6FormerDataset(
        x_file=os.path.join(data_dir, "X_val.npy"),
        y_file=os.path.join(data_dir, "y_val.npy"),
        indices_names=['txx', 'tnn', 'cdd', 'cwd', 'r95p', 'sdii'],
        target_spatial=SPATIAL_SIZE, augment=False
    )
    test_dataset = P6FormerDataset(
        x_file=os.path.join(data_dir, "X_test.npy"),
        y_file=os.path.join(data_dir, "y_test.npy"),
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
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=PIN_MEMORY)

    model = P6Former(
        time_steps=TIME_STEPS, in_vars=VARIABLES, num_indices=NUM_INDICES,
        d_model=D_MODEL, n_heads=N_HEADS, num_layers=NUM_LAYERS,
        d_ff=D_FF, dropout=DROPOUT, spatial_size=SPATIAL_SIZE
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 模型参数量: {total_params / 1e6:.2f} M")

    physics_loss_fn = PhysicsLoss(enabled=ENABLED_CONSTRAINTS, var_mean=var_mean, var_std=var_std).to(device)
    log_vars = nn.Parameter(torch.zeros(6, device=device))

    optimizer = AdamW([
        {'params': model.parameters(), 'lr': LEARNING_RATE},
        {'params': [log_vars], 'lr': LEARNING_RATE * 0.1},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print("🚀 开始训练...")
    best_val_loss = float('inf')
    patience_counter = 0
    phys_history = []

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
                # 🔧 修复：初始化为 tensor 类型的 0
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
        avg_task = total_task_loss / len(train_loader)
        avg_phys = total_phys_loss / len(train_loader)

        #验证
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X = normalize(X, var_mean, var_std).to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred_y, _ = model(X)
                loss = nn.MSELoss()(pred_y, y)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step()

        print(
            f"Epoch {epoch + 1:3d} | Train: {avg_train_loss:.4f} (task={avg_task:.4f}, phys={avg_phys:.4f}, warmup={warmup_factor:.2f}) | Val: {avg_val_loss:.4f}")

        if USE_PHYSICS and epoch >= WARMUP_EPOCHS and (epoch % 5 == 0 or epoch == WARMUP_EPOCHS):
            with torch.no_grad():
                X_diag, _ = next(iter(val_loader))
                X_diag = normalize(X_diag, var_mean, var_std).to(device)
                _, aux_diag = model(X_diag)
                phys_diag = physics_loss_fn(aux_diag)
            print("   🔍 物理约束项原始量级:")
            diag_dict = {}
            for k, v in phys_diag.items():
                print(f"      {k:20s}: {v.item():.4e}")
                diag_dict[k] = v.item()
            print(f"      log_vars: {log_vars.detach().cpu().numpy().round(4)}")
            phys_history.append({'epoch': epoch, **diag_dict})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_model.pth"))
            print(f"   ✅ 保存最佳模型 (Val Loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"⏹️ 早停触发！最佳 Val Loss: {best_val_loss:.4f}")
                break

    print("🎉 训练完成！")

    print("\n📊 加载最佳模型在测试集上评估...")
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, "best_model.pth")))
    model.eval()

    test_loss_sum = 0.0
    per_index_se = torch.zeros(NUM_INDICES, device=device)
    per_index_n = 0
    spatial_mean_mse_sum = 0.0
    n_batches = 0
    index_names = ['TXx', 'TNn', 'CDD', 'CWD', 'R95p', 'SDII']

    with torch.no_grad():
        for X, y in test_loader:
            X = normalize(X, var_mean, var_std).to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred_y, _ = model(X)

            loss = nn.MSELoss()(pred_y, y)
            test_loss_sum += loss.item()

            se = (pred_y - y) ** 2
            per_index_se += se.sum(dim=(0, 2, 3))
            per_index_n += se.shape[0] * se.shape[2] * se.shape[3]

            pred_spatial = pred_y.mean(dim=(2, 3))
            y_spatial = y.mean(dim=(2, 3))
            spatial_mean_mse_sum += ((pred_spatial - y_spatial) ** 2).mean().item()
            n_batches += 1

    avg_test_loss = test_loss_sum / len(test_loader)
    per_index_mse = (per_index_se / per_index_n).cpu().numpy()
    avg_spatial_mse = spatial_mean_mse_sum / n_batches

    print(f"\n🧪 测试集评估结果:")
    print(f"   ① 逐格点 MSE: {avg_test_loss:.4f}")
    print(f"   ② 空间平均后 MSE: {avg_spatial_mse:.4f}")
    print(f"   ③ 每指数逐格点 MSE / RMSE:")
    for name, mse_val in zip(index_names, per_index_mse):
        print(f"      {name:6s}: MSE={mse_val:.4f}, RMSE={np.sqrt(mse_val):.4f}")

    # 保存实验摘要
    summary = {
        'experiment': EXPERIMENT_NAME,
        'seed': SEED,
        'use_physics': USE_PHYSICS,
        'use_augment': USE_AUGMENT,
        'enabled_constraints': ENABLED_CONSTRAINTS,
        'physics_scale': PHYSICS_SCALE,
        'warmup_epochs': WARMUP_EPOCHS,
        'best_val_loss': best_val_loss,
        'test_loss_gridpoint_mse': avg_test_loss,
        'test_loss_spatialmean_mse': avg_spatial_mse,
        'per_index_mse': per_index_mse,
        'model_params_M': total_params / 1e6,
        'var_mean': var_mean,
        'var_std': var_std,
    }
    summary_path = os.path.join(MODEL_SAVE_DIR, "summary.npy")
    np.save(summary_path, summary)
    print(f"\n📄 实验摘要已保存: {summary_path}")

    if phys_history:
        phys_diag_path = os.path.join(MODEL_SAVE_DIR, "physics_diagnostics.npy")
        np.save(phys_diag_path, phys_history)
        print(f"📄 物理诊断历史已保存: {phys_diag_path}")


if __name__ == "__main__":
    main()