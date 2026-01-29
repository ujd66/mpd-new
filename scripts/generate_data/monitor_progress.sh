#!/bin/bash
# 监控 Rizon10s 轨迹生成进度
# 使用方法: ./monitor_progress.sh

DATA_DIR="../../data_trajectories/EnvSpheres3D-RobotRizon10s-joint_joint-one-RRTConnect"

echo "========================================"
echo "   Rizon10s 轨迹生成进度监控"
echo "========================================"
echo ""

# 检查目录是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ 数据目录不存在: $DATA_DIR"
    echo "   数据生成可能尚未开始"
    exit 1
fi

# 统计已生成的 HDF5 文件
NUM_FILES=$(find "$DATA_DIR" -name "dataset.hdf5" 2>/dev/null | wc -l)
echo "📊 已生成批次数: $NUM_FILES"

# 统计总轨迹数
if command -v python3 &> /dev/null; then
    TOTAL_TRAJS=$(python3 << EOF
import h5py
from pathlib import Path
import sys

data_dir = Path("$DATA_DIR")
total = 0
try:
    for hdf5_file in data_dir.rglob("dataset.hdf5"):
        try:
            with h5py.File(hdf5_file, 'r') as f:
                total += f.attrs.get('num_trajectories_generated', 0)
        except:
            pass
    print(total)
except Exception as e:
    print(0)
EOF
)
    
    echo "✅ 已生成轨迹数: ${TOTAL_TRAJS:=0} / 1,000,000"
    
    if [ "$TOTAL_TRAJS" -gt 0 ]; then
        PROGRESS=$(python3 -c "print(f'{$TOTAL_TRAJS / 10000.0:.2f}%')")
        echo "📈 完成进度: $PROGRESS"
        
        # 估算剩余时间（假设恒定速率）
        if [ "$TOTAL_TRAJS" -gt 1000 ]; then
            # 简单估算
            echo ""
            echo "💡 提示: 如需详细统计，请运行 python3 detailed_stats.py"
        fi
    fi
else
    echo "⚠️  需要 python3 来计算详细统计"
fi

echo ""
echo "📁 数据目录大小:"
du -sh "$DATA_DIR" 2>/dev/null || echo "  无法计算"

echo ""
echo "🔄 最近修改的文件:"
find "$DATA_DIR" -name "dataset.hdf5" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -3 | while read timestamp file; do
    date -d @${timestamp%.*} "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "未知时间"
    echo "   $file"
done

echo ""
echo "========================================"
echo "按 Ctrl+C 退出，或等待10秒自动刷新..."
sleep 10
exec "$0"  # 循环监控
