import os
import numpy as np
import pickle
from glob import glob
from tqdm import tqdm
import xarray as xr

#配置
X_DIR = r"E:\CBJdata\data2\samples_7d"
OUTPUT_DIR = r"E:\CBJdata\data2\split_7d"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_YEARS = list(range(2000, 2016))
VAL_YEARS   = list(range(2016, 2019))
TEST_YEARS  = list(range(2019, 2025))

INDICES_NAMES = ['txx', 'tnn', 'cdd', 'cwd', 'r95p', 'sdii']

#加载样本
def get_file_for_year(year):
    files = glob(os.path.join(X_DIR, f"samples_7d_{year}.nc"))
    if not files:
        raise FileNotFoundError(f"未找到 {year} 年的样本文件")
    return files[0]

def load_year_samples(year):
    file_path = get_file_for_year(year)
    ds = xr.open_dataset(file_path)
    X = ds['X'].values          # (N, lat, lon, time, var)
    y = ds['y'].values          # (N, 6, lat, lon)
    ds.close()
    return X, y

def collect_years(years, desc="加载中"):
    X_list, y_list = [], []
    for year in tqdm(years, desc=desc):
        X, y = load_year_samples(year)
        X_list.append(X)
        y_list.append(y)
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return X_all, y_all

#执行划分
print("📦 开始构建 7 天预测数据集...")

X_train, y_train = collect_years(TRAIN_YEARS, "训练集")
X_val, y_val = collect_years(VAL_YEARS, "验证集")
X_test, y_test = collect_years(TEST_YEARS, "测试集")

#保存
print("\n💾 保存数据集...")
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

np.save(os.path.join(OUTPUT_DIR, "indices_names.npy"), np.array(INDICES_NAMES))

#统计
print("\n" + "="*60)
print("✅ 7 天预测数据集构建完成！")
print(f"训练集: X {X_train.shape}, y {y_train.shape}")
print(f"验证集: X {X_val.shape}, y {y_val.shape}")
print(f"测试集: X {X_test.shape}, y {y_test.shape}")
print(f"📌 指数顺序: {INDICES_NAMES}")
print("="*60)