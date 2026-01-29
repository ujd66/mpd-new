"""
详细统计脚本 - 分析 Rizon10s 轨迹生成情况
使用方法: python detailed_stats.py
"""

import h5py
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time

DATA_DIR = Path("../../data_trajectories/EnvSpheres3D-RobotRizon10s-joint_joint-one-RRTConnect")
TARGET_TRAJECTORIES = 1_000_000


def analyze_datasets():
    """分析所有 HDF5 数据集"""
    print("=" * 80)
    print("Rizon10s 轨迹生成详细统计")
    print("=" * 80)
    print()

    if not DATA_DIR.exists():
        print(f"❌ 数据目录不存在: {DATA_DIR}")
        return

    hdf5_files = list(DATA_DIR.rglob("dataset.hdf5"))

    if not hdf5_files:
        print("⚠️  未找到任何数据文件")
        return

    print(f"📂 找到 {len(hdf5_files)} 个数据文件")
    print()

    total_desired = 0
    total_generated = 0
    file_sizes = []
    timestamps = []
    success_rates = []

    print("正在分析数据文件...")
    for hdf5_file in hdf5_files:
        try:
            with h5py.File(hdf5_file, "r") as f:
                desired = f.attrs.get("num_trajectories_desired", 0)
                generated = f.attrs.get("num_trajectories_generated", 0)

                total_desired += desired
                total_generated += generated

                if desired > 0:
                    success_rates.append(generated / desired * 100)

                file_sizes.append(hdf5_file.stat().st_size / (1024 * 1024))  # MB
                timestamps.append(hdf5_file.stat().st_mtime)
        except Exception as e:
            print(f"  ⚠️  无法读取 {hdf5_file.name}: {e}")

    # 统计汇总
    print()
    print("=" * 80)
    print("📊 统计汇总")
    print("=" * 80)
    print(f"已生成轨迹数:     {total_generated:>12,} / {TARGET_TRAJECTORIES:,}")
    print(f"完成进度:         {total_generated / TARGET_TRAJECTORIES * 100:>11.2f}%")
    print(f"目标轨迹数:       {total_desired:>12,}")
    print(f"平均成功率:       {np.mean(success_rates):>11.2f}%" if success_rates else "N/A")
    print()

    # 文件大小统计
    total_size_mb = sum(file_sizes)
    print(f"数据总大小:       {total_size_mb:>11.2f} MB")
    print(
        f"预计最终大小:     {total_size_mb / total_generated * TARGET_TRAJECTORIES if total_generated > 0 else 0:>11.2f} MB"
    )
    print()

    # 时间估算
    if len(timestamps) > 1 and total_generated > 0:
        timestamps_sorted = sorted(timestamps)
        elapsed_time = timestamps_sorted[-1] - timestamps_sorted[0]

        if elapsed_time > 0:
            rate = total_generated / elapsed_time  # 轨迹/秒
            remaining = TARGET_TRAJECTORIES - total_generated

            print("=" * 80)
            print("⏱️  时间估算")
            print("=" * 80)
            print(f"已用时间:         {timedelta(seconds=int(elapsed_time))}")
            print(f"生成速率:         {rate * 3600:>11.1f} 轨迹/小时")
            print(f"预计剩余时间:     {timedelta(seconds=int(remaining / rate))}")
            print(f"预计完成时间:     {datetime.now() + timedelta(seconds=remaining / rate):%Y-%m-%d %H:%M:%S}")
            print()

    # 最近活动
    if timestamps:
        latest_time = max(timestamps)
        time_since_last = time.time() - latest_time

        print("=" * 80)
        print("🔄 最近活动")
        print("=" * 80)
        print(f"最后更新:         {datetime.fromtimestamp(latest_time):%Y-%m-%d %H:%M:%S}")
        print(f"距今:             {timedelta(seconds=int(time_since_last))}")

        if time_since_last > 3600:  # 1小时
            print("⚠️  数据生成可能已停止或速度很慢")
        print()

    print("=" * 80)


if __name__ == "__main__":
    try:
        analyze_datasets()
    except KeyboardInterrupt:
        print("\n\n已中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()
