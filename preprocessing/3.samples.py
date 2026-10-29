import xarray as xr
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

#配置
cmfd_dir = Path(r"E:\CBJdata\data2\CMFD_0p25_v2")
extreme_dir = Path(r"E:\CBJdata\data2\daily_extreme")
output_dir = Path(r"E:\CBJdata\data2\samples_7d")
output_dir.mkdir(exist_ok=True)

INPUT_DAYS = 30
OUTPUT_DAYS = 7
STRIDE = 7
TARGET_SPATIAL = (80, 140)   # 统一下采样分辨率

#读取数据
print("📂 加载气象数据...")
cmfd_files = sorted(cmfd_dir.glob("*.nc"))
ds_cmfd = xr.open_mfdataset(cmfd_files, combine='by_coords', parallel=False,
                            engine='netcdf4', chunks={'time': 365})

print("📂 加载逐日极端指数...")
extreme_files = sorted(extreme_dir.glob("daily_extreme_*.nc"))
ds_extreme = xr.open_mfdataset(extreme_files, combine='by_coords', parallel=False,
                               engine='netcdf4', chunks={'time': 365})

print(f"   气象数据: {ds_cmfd.time.min().values} ~ {ds_cmfd.time.max().values}")
print(f"   极端指数: {ds_extreme.time.min().values} ~ {ds_extreme.time.max().values}")

if ds_cmfd.prec.units in ['kg m-2 s-1', 'kg/m2/s']:
    ds_cmfd['prec'] = ds_cmfd.prec * 86400

var_list = ['tmax', 'tmin', 'prec', 'wind', 'pres', 'srad', 'lrad', 'shum']
extreme_vars = ['txx_7d', 'tnn_7d', 'cdd_7d', 'cwd_7d', 'r95p_7d', 'sdii_7d']

#下采样函数
def downsample(arr, target=(80, 140)):
    """将空间场平均池化到目标分辨率"""
    if arr.shape[:2] == target:
        return arr
    lat_old, lon_old = arr.shape[:2]
    lat_new, lon_new = target
    lat_step = lat_old // lat_new
    lon_step = lon_old // lon_new
    lat_cut = lat_step * lat_new
    lon_cut = lon_step * lon_new
    arr = arr[:lat_cut, :lon_cut]
    arr = arr.reshape(lat_new, lat_step, lon_new, lon_step)
    arr = arr.mean(axis=(1, 3))
    return arr

#按年份构建样本
print(f"\n📅 处理年份: 2000 ~ 2024")

for year in tqdm(range(2000, 2025), desc="处理年份"):
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year}-12-31")
    input_start = year_start - pd.Timedelta(days=INPUT_DAYS + 10)

    try:
        ds_year_cmfd = ds_cmfd.sel(time=slice(input_start, year_end))
        ds_year_extreme = ds_extreme.sel(time=slice(year_start, year_end))
    except Exception as e:
        print(f"   ⚠️ {year} 年数据选取失败: {e}")
        continue

    sample_dates = pd.date_range(
        start=year_start + pd.Timedelta(days=INPUT_DAYS),
        end=year_end,
        freq=f"{STRIDE}D"
    )

    X_samples, y_samples, valid_dates = [], [], []

    for d in sample_dates:
        # 输入：d 之前的 30 天
        input_slice = ds_year_cmfd.sel(
            time=slice(d - pd.Timedelta(days=INPUT_DAYS), d)
        )
        if len(input_slice.time) < INPUT_DAYS:
            continue

        #先下采样到统一分辨率
        X_list = []
        for var in var_list:
            data = input_slice[var].values  # (30, lat, lon)
            # 对每个时间步做空间下采样
            data_ds = np.stack([downsample(data[t], TARGET_SPATIAL)
                                for t in range(data.shape[0])], axis=0)
            X_list.append(data_ds)          # (30, 80, 140)

        X = np.stack(X_list, axis=-1)       # (30, 80, 140, 8)
        X = np.transpose(X, (1, 2, 0, 3))   # (80, 140, 30, 8)
        X = np.where(np.isnan(X), 0.0, X)

        # 输出：d 对应的 7 天极端指数
        try:
            extreme_slice = ds_year_extreme.sel(time=d)
        except KeyError:
            continue

        y_list = []
        for var in extreme_vars:
            data = extreme_slice[var].values
            if data.ndim == 3:
                data = data.mean(axis=0)
            data = downsample(data, TARGET_SPATIAL)
            data = np.where(np.isnan(data), 0.0, data)
            y_list.append(data)
        y = np.stack(y_list, axis=0)        # (6, 80, 140)

        X_samples.append(X)
        y_samples.append(y)
        valid_dates.append(d)

    if not X_samples:
        print(f"   ⚠️ {year} 年无有效样本")
        continue

    X_samples = np.array(X_samples, dtype=np.float32)
    y_samples = np.array(y_samples, dtype=np.float32)
    print(f"   ✅ {year} 年样本数: {len(X_samples)}, X {X_samples.shape}, y {y_samples.shape}")

    # 保存
    out_file = output_dir / f"samples_7d_{year}.nc"
    ds_out = xr.Dataset({
        'X': (('sample', 'lat', 'lon', 'time', 'var'), X_samples),
        'y': (('sample', 'index', 'lat', 'lon'), y_samples),
        'date': (('sample',), np.array(valid_dates, dtype='datetime64[ns]')),
    })
    ds_out.to_netcdf(out_file)

print("\n🎉 7 天预测样本构建完成！")