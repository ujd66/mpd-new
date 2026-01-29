# Rizon10s 100万条轨迹生成方案

## 硬件配置
- **CPU核心数**: 32核
- **实际测试性能**: 100条轨迹 / 31.617秒 = **3.16条/秒**

## 时间估算

### 实际性能数据 ⚡
- **单条轨迹平均时间**: 0.316秒
- **生成速率**: 11,400条/小时

### 方案对比

| 方案 | 并行度 | 预计时间 | 说明 |
|------|--------|---------|------|
| 单线程 | 1 | 3.66天 | 不可行 |
| **推荐：32核并行** | 32 | **4-6小时** | ✅ 最佳方案！ |

### 计算公式
- 单线程总时间: 1,000,000轨迹 × 0.316秒 = 316,000秒 ≈ 3.66天
- 32核理论并行: 3.66天 ÷ 32 = 2.75小时
- **实际预估**（含I/O和并行开销）: **4-6小时** 🎉

---

## 推荐方案：分批并行生成

### 方案A：使用 launch_generate_trajectories.py（推荐）

**优点**：
- ✅ 自动管理多批次作业
- ✅ 失败自动重试
- ✅ 进度可追踪

**配置修改**：

```python
# 在 launch_generate_trajectories.py 中添加 Rizon10s 配置
exp_config("EnvSpheres3D", "RobotRizon10s", 1000000, 1, 0.02, "RRTConnect", 10.0, 38, 5, False, None),
```

**使用方法**：

1. **修改配置** (`launch_generate_trajectories.py` 第76行附近):
   ```python
   configs_d = {
       "one": [
           # 添加 Rizon10s 配置
           exp_config("EnvSpheres3D", "RobotRizon10s", 1000000, 1, 0.02, "RRTConnect", 10.0, 38, 5, False, None),
       ],
   }
   ```

2. **设置并行度** (第33行):
   ```python
   N_EXPS_IN_PARALLEL = 32  # 使用全部32核
   ```

3. **设置每批任务数** (第131行):
   ```python
   N_TASKS_PER_EXPERIMENT = 5000  # 每批生成5000条轨迹（基于实际性能优化）
   ```

4. **运行**：
   ```bash
   cd scripts/generate_data
   python launch_generate_trajectories.py
   ```

**预期行为**：
- 自动分成 200批 × 5000轨迹
- 每批在32核上并行运行（约需1-1.5分钟/批）
- 总时长约 **4-6小时**
- 自动保存到 `data_trajectories/EnvSpheres3D-RobotRizon10s-joint_joint-one-RRTConnect/`

---

### 方案B：手动批次脚本（更灵活）

创建自定义批处理脚本：

```bash
#!/bin/bash
# generate_rizon10s_million.sh

TOTAL_TASKS=1000000
BATCH_SIZE=10000
N_JOBS=32

for START_ID in $(seq 0 $BATCH_SIZE $((TOTAL_TASKS - 1))); do
    echo "Starting batch from task $START_ID..."
    
    python generate_trajectories.py \
        --env_id EnvSpheres3D \
        --robot_id RobotRizon10s \
        --start_task_id $START_ID \
        --num_tasks $BATCH_SIZE \
        --num_trajectories_per_task 1 \
        --n_parallel_jobs $N_JOBS \
        --debug false
    
    echo "Batch completed: tasks $START_ID to $((START_ID + BATCH_SIZE))"
done

echo "All 1 million trajectories generated!"
```

**运行**：
```bash
chmod +x generate_rizon10s_million.sh
./generate_rizon10s_million.sh
```

---

## 优化建议

### 1. 提高成功率
```python
# 在 generate_trajectories.py 中
planner_allowed_time: float = 15.0  # 增加到15秒（从10秒）
```

### 2. 磁盘空间预估
- 每条轨迹约 2KB（128个waypoints × 7 DOF）
- 100万条 = **2GB** 数据
- 建议预留 **5-10GB** 空间

### 3. 实时监控
```bash
# 监控进度
watch -n 10 "find data_trajectories -name 'dataset.hdf5' -exec wc -c {} + | tail -1"

# 查看已生成轨迹数
python -c "import h5py; print(sum([h5py.File(f, 'r').attrs['num_trajectories_generated'] for f in Path('data_trajectories').rglob('dataset.hdf5')]))"
```

### 4. 断点续传
如果中断，修改 `start_task_id` 继续：
```python
--start_task_id 500000  # 从50万条继续
```

---

## 后处理流程

生成完成后：

### 1. 合并数据集
```bash
cd scripts/generate_data
python post_process_generated_dataset.py \
    ../../data_trajectories/EnvSpheres3D-RobotRizon10s-joint_joint-one-RRTConnect
```

### 2. 验证数据质量
```python
import h5py
import numpy as np

# 检查数据集
with h5py.File('dataset_merged.hdf5', 'r') as f:
    print(f"Total trajectories: {len(f['sol_path'])}")
    print(f"Task IDs: {len(np.unique(f['task_id']))}")
    print(f"Keys: {list(f.keys())}")
```

### 3. 可视化采样
```bash
python visualize_trajectories.py \
    ../../data_trajectories/EnvSpheres3D-RobotRizon10s-joint_joint-one-RRTConnect
```

---

## 时间线规划

假设使用32核并行（**基于实际测试数据**）：

| 阶段 | 时间 | 说明 |
|------|------|------|
| **阶段1：数据生成** | 4-6小时 | 100万条轨迹（实测：3.16条/秒） |
| **阶段2：后处理** | 1-2小时 | 合并、压缩 |
| **阶段3：验证** | 30分钟 | 质量检查 |
| **总计** | **~6-9小时** | ✅ 当天完成！ |

---

## 故障排查

### 问题1：进程被杀（OOM）
**解决**：减小批次大小
```python
N_TASKS_PER_EXPERIMENT = 500  # 从1000降到500
```

### 问题2：PyBullet崩溃
**解决**：降低并行度
```python
N_EXPS_IN_PARALLEL = 16  # 从32降到16
```

### 问题3：磁盘满
**解决**：定期合并并删除临时文件
```bash
# 每10万条合并一次
python post_process_generated_dataset.py <dir> --delete_temp
```

---

## 最终推荐

**最佳实践组合**（基于实际性能）：

1. ✅ 使用 **方案A**（launch_generate_trajectories.py）或 **专用脚本**（launch_rizon10s_million.py）
2. ✅ **32核全并行**
3. ✅ **每批5000条**轨迹（优化后）
4. ✅ 预计 **4-6小时**完成 🚀
5. ✅ 实时监控进度（使用 `monitor_progress.sh`）

**立即开始命令**：
```bash
cd /home/bochu/code/mpd/mpd-splines-public/scripts/generate_data

# 1. 修改 launch_generate_trajectories.py（添加Rizon10s配置）
# 2. 运行
python launch_generate_trajectories.py
```
