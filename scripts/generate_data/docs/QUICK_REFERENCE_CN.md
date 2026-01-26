# 快速参考

## 常用命令

```bash
# 生成小规模测试数据
python generate_trajectories.py

# 并行生成大规模数据
python launch_generate_trajectories.py

# 合并数据集
python post_process_generated_dataset.py --data_dir ./data/env-robot/

# 数据增强（翻转路径）
python flip_solution_paths.py

# 可视化
python visualize_trajectories.py --data_dir ./data/env-robot/
```

## 关键配置

**generate_trajectories.py**
- 第 449-450 行：环境和机器人
- 第 452-454 行：轨迹数量

**flip_solution_paths.py**
- 第 56 行：数据集路径

## 数据位置

- 单次生成 → `./data/env-robot/dataset.hdf5`
- 批量生成 → `../../data_trajectories/`
- 合并后 → `dataset_merged.hdf5`
- 增强后 → `dataset_merged_doubled.hdf5`
