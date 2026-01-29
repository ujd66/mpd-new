"""
轨迹质量评估脚本
评估生成的 Rizon10s 轨迹数据集的质量
"""

import sys
import math
import fractions

# Fix for networkx < 2.2 compatibility with Python 3.9+
if not hasattr(fractions, "gcd"):
    fractions.gcd = math.gcd

# 添加路径
sys.path.insert(0, "/home/bochu/code/mpd/mpd-splines-public")

from mpd.utils.patches import numpy_monkey_patch

numpy_monkey_patch()

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


from torch_robotics.robots import RobotRizon10s
from torch_robotics.environments import EnvSpheres3D
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
import torch


def evaluate_trajectory_quality(dataset_path):
    """评估轨迹质量"""

    print("=" * 80)
    print(f"轨迹质量评估报告")
    print("=" * 80)
    print(f"数据集路径: {dataset_path}")
    print()

    # 加载数据
    with h5py.File(dataset_path, "r") as f:
        print("📊 数据集基本信息")
        print("-" * 80)
        print(f"数据集键: {list(f.keys())}")
        print(f"目标轨迹数: {f.attrs.get('num_trajectories_desired', 'N/A')}")
        print(f"实际生成数: {f.attrs.get('num_trajectories_generated', 'N/A')}")
        print()

        # 读取轨迹数据
        trajectories = f["sol_path"][:]  # (N, timesteps, DOF)
        task_ids = f["task_id"][:]

        if "solution_times" in f:
            solution_times = f["solution_times"][:]
        else:
            solution_times = None

        if "num_nodes" in f:
            num_nodes = f["num_nodes"][:]
        else:
            num_nodes = None

    num_trajs, timesteps, dof = trajectories.shape
    print(f"轨迹形状: {trajectories.shape}")
    print(f"  - 轨迹数量: {num_trajs}")
    print(f"  - 时间步数: {timesteps}")
    print(f"  - 自由度(DOF): {dof}")
    print()

    # 初始化机器人和环境
    print("🤖 加载 Rizon10s 机器人和环境...")
    robot = RobotRizon10s(tensor_args=DEFAULT_TENSOR_ARGS)
    env = EnvSpheres3D(tensor_args=DEFAULT_TENSOR_ARGS)
    print()

    # 质量指标
    print("=" * 80)
    print("📈 轨迹质量指标")
    print("=" * 80)

    # 1. 路径长度分析
    print("\n1️⃣  路径长度分析")
    print("-" * 80)
    path_lengths = []
    for traj in trajectories:
        # 计算每段的关节空间距离
        distances = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        total_length = np.sum(distances)
        path_lengths.append(total_length)

    path_lengths = np.array(path_lengths)
    print(f"平均路径长度: {np.mean(path_lengths):.4f} rad")
    print(f"最短路径: {np.min(path_lengths):.4f} rad")
    print(f"最长路径: {np.max(path_lengths):.4f} rad")
    print(f"标准差: {np.std(path_lengths):.4f} rad")

    # 2. 关节限制检查
    print("\n2️⃣  关节限制检查")
    print("-" * 80)
    q_min = robot.q_pos_min.cpu().numpy()
    q_max = robot.q_pos_max.cpu().numpy()

    violations_min = np.sum(trajectories < q_min, axis=(0, 1))
    violations_max = np.sum(trajectories > q_max, axis=(0, 1))
    total_violations = np.sum(violations_min) + np.sum(violations_max)

    print(f"关节限制违反总数: {total_violations}")
    if total_violations > 0:
        print(f"  ⚠️  存在关节限制违反！")
        for i in range(dof):
            if violations_min[i] > 0 or violations_max[i] > 0:
                print(f"  关节 {i+1}: 最小违反 {violations_min[i]}, 最大违反 {violations_max[i]}")
    else:
        print(f"  ✅ 所有轨迹均满足关节限制")

    # 3. 速度分析
    print("\n3️⃣  速度分析")
    print("-" * 80)
    velocities = np.diff(trajectories, axis=1)  # (N, timesteps-1, DOF)
    max_velocities = np.max(np.abs(velocities), axis=1)  # (N, DOF)

    print(f"各关节平均最大速度:")
    for i in range(dof):
        print(f"  关节 {i+1}: {np.mean(max_velocities[:, i]):.4f} rad/timestep")

    # 检查速度限制（如果有的话）
    if robot.dq_max is not None:
        dq_max = robot.dq_max.cpu().numpy()
        # 假设timestep = 1/128秒（从interpolate_num推断）
        dt = 1.0 / 128.0
        velocity_violations = np.sum(np.abs(velocities) > dq_max * dt)
        print(f"\n速度限制违反数: {velocity_violations}")
        if velocity_violations == 0:
            print(f"  ✅ 所有速度均在限制范围内")

    # 4. 加速度平滑度
    print("\n4️⃣  加速度和平滑度分析")
    print("-" * 80)
    accelerations = np.diff(velocities, axis=1)  # (N, timesteps-2, DOF)
    max_accelerations = np.max(np.abs(accelerations), axis=1)  # (N, DOF)

    print(f"各关节平均最大加速度:")
    for i in range(dof):
        print(f"  关节 {i+1}: {np.mean(max_accelerations[:, i]):.4f} rad/timestep²")

    # 计算抖动（jerk）
    jerks = np.diff(accelerations, axis=1)
    mean_jerk = np.mean(np.abs(jerks))
    print(f"\n平均抖动(jerk): {mean_jerk:.6f} rad/timestep³")

    # 5. 规划效率
    if solution_times is not None:
        print("\n5️⃣  规划效率分析")
        print("-" * 80)
        print(f"平均求解时间: {np.mean(solution_times):.4f} 秒")
        print(f"最快求解: {np.min(solution_times):.4f} 秒")
        print(f"最慢求解: {np.max(solution_times):.4f} 秒")
        print(f"标准差: {np.std(solution_times):.4f} 秒")

    if num_nodes is not None:
        print(f"\n平均扩展节点数: {np.mean(num_nodes):.1f}")
        print(f"最少节点: {np.min(num_nodes)}")
        print(f"最多节点: {np.max(num_nodes)}")

    # 6. 碰撞检查（采样检查）
    print("\n6️⃣  碰撞检查 (采样检查)")
    print("-" * 80)
    sample_indices = np.random.choice(num_trajs, min(10, num_trajs), replace=False)
    collision_count = 0

    for idx in sample_indices:
        traj = trajectories[idx]
        # 检查每个waypoint
        for waypoint in traj:
            # 将 numpy 数组转换为 tensor 并移动到机器人的设备上
            q_torch = torch.from_numpy(waypoint).float().to(robot.q_pos_min.device)
            # 获取机器人碰撞球位置
            positions = robot.fk_map_collision(q_torch)

            # 检查与环境的碰撞
            if hasattr(env, "compute_cost"):
                cost = env.compute_cost(positions.unsqueeze(0))
                if cost > 0:
                    collision_count += 1
                    break

    print(f"采样 {len(sample_indices)} 条轨迹")
    print(f"检测到碰撞: {collision_count} 条")
    if collision_count == 0:
        print(f"  ✅ 采样轨迹均无碰撞")
    else:
        print(f"  ⚠️  部分轨迹可能存在碰撞")

    # 总结
    print("\n" + "=" * 80)
    print("📋 质量评估总结")
    print("=" * 80)

    quality_score = 100
    issues = []

    if total_violations > 0:
        quality_score -= 20
        issues.append("存在关节限制违反")

    if collision_count > 0:
        quality_score -= 30
        issues.append("可能存在碰撞")

    if mean_jerk > 0.1:
        quality_score -= 10
        issues.append("轨迹不够平滑")

    print(f"\n总体质量评分: {quality_score}/100")

    if issues:
        print(f"\n发现的问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print(f"\n✅ 轨迹质量优秀！")

    print("\n" + "=" * 80)

    # 可视化
    generate_plots(trajectories, path_lengths, solution_times, num_nodes)

    return {
        "num_trajectories": num_trajs,
        "path_lengths": path_lengths,
        "total_violations": total_violations,
        "collision_count": collision_count,
        "quality_score": quality_score,
    }


def generate_plots(trajectories, path_lengths, solution_times, num_nodes):
    """生成可视化图表"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 路径长度分布
    axes[0, 0].hist(path_lengths, bins=30, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Path Length (rad)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of Path Lengths")
    axes[0, 0].axvline(np.mean(path_lengths), color="red", linestyle="--", label=f"Mean: {np.mean(path_lengths):.2f}")
    axes[0, 0].legend()

    # 2. 示例轨迹
    sample_traj = trajectories[0]
    for i in range(min(7, sample_traj.shape[1])):  # 画前7个关节
        axes[0, 1].plot(sample_traj[:, i], label=f"Joint {i+1}")
    axes[0, 1].set_xlabel("Timestep")
    axes[0, 1].set_ylabel("Joint Position (rad)")
    axes[0, 1].set_title("Example Trajectory (First)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 求解时间分布
    if solution_times is not None:
        axes[1, 0].hist(solution_times, bins=30, edgecolor="black", alpha=0.7)
        axes[1, 0].set_xlabel("Solution Time (s)")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Distribution of Solution Times")
        axes[1, 0].axvline(
            np.mean(solution_times), color="red", linestyle="--", label=f"Mean: {np.mean(solution_times):.3f}s"
        )
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, "No solution time data", ha="center", va="center", transform=axes[1, 0].transAxes)

    # 4. 节点数分布
    if num_nodes is not None:
        axes[1, 1].hist(num_nodes, bins=30, edgecolor="black", alpha=0.7)
        axes[1, 1].set_xlabel("Number of Nodes")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Distribution of Nodes Expanded")
        axes[1, 1].axvline(np.mean(num_nodes), color="red", linestyle="--", label=f"Mean: {np.mean(num_nodes):.0f}")
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, "No node count data", ha="center", va="center", transform=axes[1, 1].transAxes)

    plt.tight_layout()

    # 保存图表
    output_path = "/home/bochu/code/mpd/mpd-splines-public/trajectory_quality_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 可视化图表已保存到: {output_path}")

    # 不显示（因为可能在终端运行）
    # plt.show()
    plt.close()


if __name__ == "__main__":
    dataset_path = "/home/bochu/code/mpd/mpd-splines-public/data/env-robot/1769418565/dataset.hdf5"

    if not Path(dataset_path).exists():
        print(f"❌ 数据集不存在: {dataset_path}")
        sys.exit(1)

    try:
        results = evaluate_trajectory_quality(dataset_path)
        print(f"\n✅ 评估完成！")
    except Exception as e:
        print(f"\n❌ 评估过程中出错: {e}")
        import traceback

        traceback.print_exc()
