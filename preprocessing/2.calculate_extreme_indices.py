import xarray as xr
import numpy as np
from pathlib import Path
import pandas as pd

# 配置路径
data_dir = Path(r"E:\CBJdata\data2\CMFD_0p25_v2")
output_dir = Path(r"E:\CBJdata\data2\daily_extreme")
output_dir.mkdir(exist_ok=True)

#打开数据集
files = sorted(data_dir.glob("*.nc"))
print(f"📂 找到 {len(files)} 个文件")

ds = xr.open_mfdataset(files, combine='by_coords', parallel=False,
                       engine='netcdf4', chunks={'time': 365})

print(f"📊 数据范围: {ds.time.min().values} ~ {ds.time.max().values}")
print(f"   网格: {ds.lat.size} × {ds.lon.size}")

if ds.prec.units in ['kg m-2 s-1', 'kg/m2/s']:
    print("⚠️ 降水单位转换为 mm/day")
    ds['prec'] = ds.prec * 86400

# 计算 95% 分位数阈值
print("\n⏳ 计算基准 95% 分位数阈值...")
pr_all = ds.prec.where(ds.prec >= 1.0)
threshold_95 = pr_all.quantile(0.95, dim='time', skipna=True).compute()
threshold_95 = threshold_95.fillna(1e6)
print(f"   ✅ 阈值计算完成，有效格点数: {int((threshold_95 < 1e5).sum().item())}")

#逐年处理
print("\n⏳ 逐年计算 7 天极端指数...")
for year in range(2000, 2025):
    print(f"\n📅 处理 {year} 年...")

    start = pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=6)
    end = pd.Timestamp(f"{year}-12-31") + pd.Timedelta(days=6)

    try:
        ds_year = ds.sel(time=slice(start, end))
    except Exception as e:
        print(f"   ⚠️ 数据选取失败: {e}")
        continue

    # ---- 温度 ----
    tmax_7d = ds_year.tmax.rolling(time=7, min_periods=7).max().shift(time=-6)
    tmin_7d = ds_year.tmin.rolling(time=7, min_periods=7).min().shift(time=-6)

    # ---- 降水 ----
    prec_7d = ds_year.prec.rolling(time=7, min_periods=7).sum().shift(time=-6)
    wet_7d = (ds_year.prec >= 1.0).rolling(time=7, min_periods=7).sum().shift(time=-6)

    # ---- SDII ----
    prec_7d_np = prec_7d.compute().values
    wet_7d_np = wet_7d.compute().values
    with np.errstate(divide='ignore', invalid='ignore'):
        sdii_7d_np = np.where(wet_7d_np > 0,
                              prec_7d_np / np.maximum(wet_7d_np, 1e-10),
                              np.nan).astype(np.float32)
    sdii_7d = xr.DataArray(sdii_7d_np, coords=prec_7d.coords,
                           dims=prec_7d.dims, name='sdii_7d')

    # ---- R95p ----
    exceed = ds_year.prec.where(ds_year.prec > threshold_95, 0.0)
    r95p_7d = exceed.rolling(time=7, min_periods=7).sum().shift(time=-6)

    # ---- CDD / CWD ----

    wet_7d_np2 = wet_7d.compute().values
    cwd_7d_np = np.where(wet_7d_np2 > 0, wet_7d_np2, 0).astype(np.float32)
    cdd_7d_np = (7.0 - wet_7d_np2).astype(np.float32)

    cwd_7d = xr.DataArray(cwd_7d_np, coords=wet_7d.coords,
                          dims=wet_7d.dims, name='cwd_7d')
    cdd_7d = xr.DataArray(cdd_7d_np, coords=wet_7d.coords,
                          dims=wet_7d.dims, name='cdd_7d')

    # float32
    result = xr.Dataset({
        'txx_7d': tmax_7d.astype(np.float32),
        'tnn_7d': tmin_7d.astype(np.float32),
        'cdd_7d': cdd_7d,
        'cwd_7d': cwd_7d,
        'r95p_7d': r95p_7d.astype(np.float32),
        'sdii_7d': sdii_7d,
    })

    # 只保留当年
    result_year = result.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))

    # 保存
    out_file = output_dir / f"daily_extreme_{year}.nc"
    result_year.compute().to_netcdf(out_file)
    print(f"   ✅ 已保存: {out_file}")

print("\n🎉 逐日极端指数计算完成！")