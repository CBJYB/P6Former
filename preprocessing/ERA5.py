
import time
from pathlib import Path
import cdsapi
from tqdm import tqdm


AREA = [55, 70, 15, 140]          # [北, 西, 南, 东] 覆盖青藏高原及周边
YEARS = list(range(2000, 2025))
MONTHS = list(range(1, 13))

OUTPUT_DIR = Path("./era5_land_temp_hourly")   # 存放下载的原始NC文件
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_RETRIES = 3
MIN_SIZE_MB = 10

def download_month(year, month, client):
    """下载一个月的小时温度数据，保存为 NetCDF 文件"""
    base = f"era5_land_t2m_hourly_{year}{month:02d}"
    nc_path = OUTPUT_DIR / f"{base}.nc"
    zip_path = OUTPUT_DIR / f"{base}.zip"

    # 如果已有NC且大小合理，跳过
    if nc_path.exists():
        sz = nc_path.stat().st_size / (1024*1024)
        if sz >= MIN_SIZE_MB:
            print(f"{year}-{month:02d}: NC已存在 ({sz:.1f} MB)，跳过")
            return True, nc_path
        else:
            nc_path.unlink()

    if zip_path.exists():
        sz = zip_path.stat().st_size / (1024*1024)
        if sz >= MIN_SIZE_MB:
            print(f"{year}-{month:02d}: ZIP已存在 ({sz:.1f} MB)，尝试转为NC")

            zip_path.unlink()
        else:
            zip_path.unlink()

    for attempt in range(1, MAX_RETRIES+1):
        try:
            print(f"{year}-{month:02d}: 下载尝试 {attempt}/{MAX_RETRIES} ...")
            # 请求数据
            client.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': ['2m_temperature'],
                    'year': str(year),
                    'month': f"{month:02d}",
                    'day': [f"{d:02d}" for d in range(1, 32)],
                    'time': [f"{h:02d}:00" for h in range(24)],
                    'area': AREA,
                    'format': 'netcdf',
                },
                str(nc_path)    # 直接保存为.nc
            )
            # 检查文件大小
            sz = nc_path.stat().st_size / (1024*1024)
            if sz < MIN_SIZE_MB:
                raise Exception(f"文件过小 ({sz:.1f} MB)")
            print(f"  ✅ 下载完成，大小 {sz:.1f} MB，保存为 {nc_path.name}")
            return True, nc_path
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            if attempt < MAX_RETRIES:
                wait = 10 * attempt
                print(f"    等待 {wait} 秒后重试...")
                time.sleep(wait)
            else:
                print(f"    放弃 {year}-{month:02d}")
                # 删除可能的不完整文件
                if nc_path.exists():
                    nc_path.unlink()
    return False, None

def main():
    print("=" * 60)
    print("ERA5-Land 逐小时 2m 温度数据批量下载 (2000-2024)")
    print(f"输出目录: {OUTPUT_DIR.absolute()}")
    print(f"空间范围: {AREA}")
    print("=" * 60)

    choice = input("\n是否删除已有文件，全部重新下载？(y/n, 默认n): ").strip().lower()
    if choice == 'y':
        for f in OUTPUT_DIR.glob("*.nc"):
            f.unlink()
        for f in OUTPUT_DIR.glob("*.zip"):
            f.unlink()
        print("已清空目录。")

    client = cdsapi.Client()
    total = len(YEARS) * len(MONTHS)
    success = 0
    fail = []

    with tqdm(total=total, desc="下载进度") as pbar:
        for year in YEARS:
            for month in MONTHS:
                ok, _ = download_month(year, month, client)
                if ok:
                    success += 1
                else:
                    fail.append(f"{year}-{month:02d}")
                pbar.update(1)

    print("\n" + "=" * 60)
    print(f"下载完成。成功: {success}/{total}")
    if fail:
        print(f"失败月份: {fail}")
        print("请手动重试失败月份，或检查网络/CDS账户状态。")
    else:
        print("✅ 全部成功！")
    print("=" * 60)

if __name__ == "__main__":
    main()