# 数据生成脚本使用说明

## 🚀 快速开始

### 生成测试数据（5条轨迹）
```bash
python generate_trajectories.py
```

### 生成大规模数据集
```bash
# 1. 并行生成（需先修改配置）
python launch_generate_trajectories.py

# 2. 合并数据集
python post_process_generated_dataset.py --data_dir ./data/env-robot/

# 3. 增强数据（可选）
python flip_solution_paths.py

# 4. 可视化
python visualize_trajectories.py --data_dir ./data/env-robot/
```

---

## 📝 脚本说明

| 脚本 | 作用 |
|------|------|
| `generate_trajectories.py` | 使用 OMPL 生成轨迹 |
| `launch_generate_trajectories.py` | 并行批量生成 |
| `post_process_generated_dataset.py` | 合并多个数据集 |
| `flip_solution_paths.py` | 翻转路径增强数据 |
| `visualize_trajectories.py` | 可视化轨迹 |

---

## ⚙️ 常用配置

**修改环境和机器人**（`generate_trajectories.py` 第 449-450 行）：
```python
env_id: str = "EnvWarehouse",    # EnvSimple2D, EnvSpheres3D
robot_id: str = "RobotPanda",    # RobotPointMass2D
```

**修改轨迹数量**（第 452-454 行）：
```python
num_tasks: int = 100,              # 任务数
num_trajectories_per_task: int = 1, # 每任务轨迹数
```

**修改路径**（`flip_solution_paths.py` 第 56 行）：
```python
PATH_TO_DATASETS = "./data/**/*dataset_merged.hdf5"
```

---

## 💾 数据保存位置

- `generate_trajectories.py` → `./data/env-robot/dataset.hdf5`
- `launch_generate_trajectories.py` → `../../data_trajectories/`
- 合并后 → `dataset_merged.hdf5`
- 增强后 → `dataset_merged_doubled.hdf5`

---

## 🔧 已修复兼容性问题

所有脚本已添加 `networkx` 2.2 与 Python 3.9+/NumPy 1.20+ 兼容性修复。

详细文档见 `docs/` 目录。
