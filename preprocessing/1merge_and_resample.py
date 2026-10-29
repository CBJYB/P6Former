import xarray as xr
import numpy as np
from pathlib import Path

#配置路径
data_dir = Path(r"E:\CBJdata\data2\deal_data")
output_dir = Path(r"E:\CBJdata\data2\CMFD_0p25_v2")
output_dir.mkdir(exist_ok=True)

# CMFD 变量文件夹列表
cmfd_vars = ['lrad', 'prec', 'pres', 'rain', 'shum', 'snow', 'srad', 'wind']

def normalize_time(ds):
    """
    将数据集的时间坐标强制转换为日期类型（精度到日），
    使用 floor('D') 向下取整到天，兼容所有 xarray 版本。
    """
    ds = ds.copy()
    ds['time'] = ds.time.dt.floor('D')   # 替代 normalize()
    return ds

#读取 CMFD 变量
print("📂 读取 CMFD 变量...")
ds_cmfd_list = []
for var in cmfd_vars:
    folder = data_dir / var
    files = sorted(folder.glob("*.nc"))
    if not files:
        print(f"   ⚠️ {var}: 文件夹为空，跳过")
        continue

    ds = xr.open_mfdataset(files, combine='by_coords', parallel=False, engine='netcdf4')
    # 确保变量名正确
    if var not in ds.data_vars:
        if len(ds.data_vars) == 1:
            old_name = list(ds.data_vars)[0]
            ds = ds.rename({old_name: var})
        else:
            print(f"   ⚠️ {var}: 变量名不唯一，跳过")
            continue
    ds = normalize_time(ds)
    ds_cmfd_list.append(ds)
    print(f"   ✅ {var}: {len(ds.time)} 天")

#读取 ERA5 温度
print("\n📂 读取 ERA5 温度...")
temp_folder = data_dir / "temp"
temp_files = sorted(temp_folder.glob("*.nc"))
if not temp_files:
    raise FileNotFoundError("temp 文件夹中未找到任何文件")
ds_temp = xr.open_mfdataset(temp_files, combine='by_coords', parallel=False, engine='netcdf4')
if 'tmax' not in ds_temp.data_vars or 'tmin' not in ds_temp.data_vars:
    raise ValueError("temp 文件中未找到 tmax/tmin 变量")
ds_temp = normalize_time(ds_temp)
print(f"   ✅ tmax/tmin: {len(ds_temp.time)} 天")

#合并所有变量
print("\n🔄 合并所有变量...")
ds_all = xr.merge(ds_cmfd_list + [ds_temp[['tmax', 'tmin']]], join='inner')
print(f"   ✅ 合并完成，包含变量: {list(ds_all.data_vars)}")
print(f"   ⏰ 时间范围: {ds_all.time.min().values} ~ {ds_all.time.max().values}")
print(f"   📆 总天数: {len(ds_all.time)}（应为 9132 天）")

#空间重采样到 0.25°
print("\n🗺️ 重采样到 0.25° 网格...")
lat_vals = ds_all.lat.values
lon_vals = ds_all.lon.values
lat_min = np.floor(lat_vals.min() / 0.25) * 0.25
lat_max = np.ceil(lat_vals.max() / 0.25) * 0.25
lon_min = np.floor(lon_vals.min() / 0.25) * 0.25
lon_max = np.ceil(lon_vals.max() / 0.25) * 0.25
new_lat = np.arange(lat_max, lat_min - 0.25, -0.25)  # 北→南
new_lon = np.arange(lon_min, lon_max + 0.25, 0.25)   # 西→东
print(f"   目标网格: {len(new_lat)} 纬度 × {len(new_lon)} 经度")

ds_resampled = ds_all.interp(
    lat=new_lat,
    lon=new_lon,
    method='linear',
    kwargs={'fill_value': np.nan}
)
print(f"   ✅ 插值完成")

#按年份保存
print("\n💾 按年份保存...")
comp = dict(zlib=True, complevel=5)  # 压缩参数，减小文件大小
encoding = {var: comp for var in ds_resampled.data_vars}

for year in range(2000, 2025):
    try:
        ds_year = ds_resampled.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        out_file = output_dir / f"CMFD_0p25_{year}.nc"
        ds_year.to_netcdf(out_file, encoding=encoding)
        print(f"   ✅ {year} 年已保存 ({len(ds_year.time)} 天)")
    except Exception as e:
        print(f"   ❌ {year} 年保存失败: {e}")

print(f"\n🎉 全部完成！输出目录: {output_dir}")