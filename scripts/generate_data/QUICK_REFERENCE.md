# Rizon10s 数据生成快速参考

## 🚀 快速开始

### 选项1: 使用专用脚本（推荐新手）
```bash
cd /home/bochu/code/mpd/mpd-splines-public/scripts/generate_data
python launch_rizon10s_million.py
```

### 选项2: 使用通用启动器（高级用户）
修改 `launch_generate_trajectories.py` 第76行后附近：
```python
exp_config("EnvSpheres3D", "RobotRizon10s", 1000000, 1, 0.02, "RRTConnect", 10.0, 38, 5, False, None),
```
然后运行：
```bash
python launch_generate_trajectories.py
```

### 选项3: 手动批次（最灵活）
```bash
# 生成第一批1万条
python generate_trajectories.py \
    --env_id EnvSpheres3D \
    --robot_id RobotRizon10s \
    --start_task_id 0 \
    --num_tasks 10000 \
    --num_trajectories_per_task 1 \
    --n_parallel_jobs 32 \
    --debug false
```

---

## 📊 监控进度

### 实时监控（自动刷新）
```bash
chmod +x monitor_progress.sh
./monitor_progress.sh
```

### 详细统计
```bash
python detailed_stats.py
```

### 快速查看
```bash
# 查看已生成文件数
find ../../data_trajectories -name "dataset.hdf5" | wc -l

# 查看数据总大小
du -sh ../../data_trajectories/EnvSpheres3D-RobotRizon10s*
```

---

## ⚙️ 性能调优

### CPU核心数调整
32核（推荐）：
```python
n_parallel_jobs = 32  # 最快
```

16核（稳定）：
```python
n_parallel_jobs = 16  # 降低内存压力
```

### 规划时间调整
```python
planner_allowed_time = 15.0  # 增加到15秒，提高复杂场景成功率
```

### 批次大小调整
```python
N_TASKS_PER_EXPERIMENT = 500  # 减小批次，降低内存占用
```

---

## 🔧 常见问题

### 进程被杀（OOM）
```bash
# 方案1: 减小批次
N_TASKS_PER_EXPERIMENT = 200

# 方案2: 降低并行度
n_parallel_jobs = 16
```

### 磁盘空间不足
```bash
# 定期合并数据
cd ../../data_trajectories/EnvSpheres3D-RobotRizon10s*
python ../../scripts/generate_data/post_process_generated_dataset.py .
```

### 想要暂停/继续
```bash
# Ctrl+C 暂停当前批次
# 记录最后的 start_task_id
# 修改启动脚本，从该ID继续
--start_task_id 500000
```

---

## 📦 数据后处理

### 合并所有批次
```bash
cd scripts/generate_data
python post_process_generated_dataset.py \
    ../../data_trajectories/EnvSpheres3D-RobotRizon10s-joint_joint-one-RRTConnect
```

### 验证数据质量
```python
import h5py
with h5py.File('dataset_merged.hdf5', 'r') as f:
    print(f"Total: {len(f['sol_path'])} trajectories")
    print(f"Keys: {list(f.keys())}")
```

### 可视化检查
```bash
python visualize_trajectories.py \
    ../../data_trajectories/EnvSpheres3D-RobotRizon10s-joint_joint-one-RRTConnect
```

---

## 📈 预期时间线

| 阶段 | 时间 | 说明 |
|------|------|------|
| 数据生成 | 3-4天 | 32核并行 |
| 后处理 | 2小时 | 合并压缩 |
| 验证 | 1小时 | 质量检查 |

---

## 💾 磁盘空间需求

- 原始数据: ~2GB
- 后处理: ~3GB
- 建议预留: **10GB**

---

## ✅ 完成检查清单

- [ ] 启动数据生成
- [ ] 定期检查进度（每天）
- [ ] 监控磁盘空间
- [ ] 数据生成完成
- [ ] 运行后处理脚本
- [ ] 验证数据质量
- [ ] 备份数据

---

查看完整文档: `RIZON10S_DATA_GENERATION_PLAN.md`
