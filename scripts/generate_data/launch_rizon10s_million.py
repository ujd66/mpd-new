"""
快速启动脚本：生成100万条 Rizon10s 轨迹
使用方法：python launch_rizon10s_million.py
"""

import collections
import os
import socket
import sys
import time
from pathlib import Path

# 设置环境变量（确保子进程能找到 mpd 模块和 OMPL）
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OMPL_BUILD_DIR = PROJECT_ROOT / "deps/pybullet_ompl/ompl/build/Release"

# 添加项目根目录到 PYTHONPATH
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 设置环境变量（会被子进程继承）
os.environ["PYTHONPATH"] = (
    f"{OMPL_BUILD_DIR}/lib:{PROJECT_ROOT}/deps/pybullet_ompl/ompl/py-bindings:{PROJECT_ROOT}:{os.environ.get('PYTHONPATH', '')}"
)
os.environ["LD_LIBRARY_PATH"] = f"{OMPL_BUILD_DIR}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

from experiment_launcher import Launcher
from experiment_launcher.utils import is_local, get_slurm_jobs_in_queue

# 配置确认
response = input("将生成100万条 EnvSpheres3D-RobotRizon10s 轨迹，预计4-6小时完成。继续？ (yes/no): ").lower()
if response not in ["yes", "y"]:
    exit(1)

########################################################################################################################
# 系统配置

hostname = socket.gethostname()
LOCAL = is_local()
TEST = False
USE_CUDA = False

# 并行配置
# 策略：每次运行1个实验，但该实验内部使用32核并行处理
N_EXPS_IN_PARALLEL = 1  # 同时运行的实验数（避免内存爆炸）
N_PARALLEL_JOBS = 32  # 每个实验内部的并行核心数
N_CORES = 32  # 分配给每个实验的核心数
MEMORY_SINGLE_JOB = 3000
MEMORY_PER_CORE = MEMORY_SINGLE_JOB

PARTITION = None
GRES = None
CONDA_ENV = "mpd-splines-public"

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

########################################################################################################################
# 轨迹生成配置

exp_config = collections.namedtuple(
    "config",
    "env_id robot_id"
    " num_tasks num_trajectories_per_task"
    " min_distance_robot_env"
    " planner planner_allowed_time"
    " bspline_num_control_points bspline_degree"
    " sample_joint_position_goals_with_same_ee_pose"
    " cfg_file",
)

# Rizon10s 配置
RIZON10S_CONFIG = exp_config(
    env_id="EnvSpheres3D",
    robot_id="RobotRizon10s",
    num_tasks=1000000,  # 100万条轨迹
    num_trajectories_per_task=1,  # 每个任务1条
    min_distance_robot_env=0.02,
    planner="RRTConnect",
    planner_allowed_time=10.0,
    bspline_num_control_points=38,
    bspline_degree=5,
    sample_joint_position_goals_with_same_ee_pose=False,
    cfg_file=None,
)

# 每批任务数（基于实际性能优化）
N_TASKS_PER_EXPERIMENT = 5000  # 每批5000条，共200批

########################################################################################################################
# 启动作业

config = RIZON10S_CONFIG
# 默认从头开始生成数据
start_task_id = 0
n_tasks_to_process_total = config.num_tasks
n_job = 0

print(f"\n{'='*80}")
print(f"开始生成 Rizon10s 轨迹数据")
print(f"{'='*80}")
print(f"环境: {config.env_id}")
print(f"机器人: {config.robot_id}")
print(f"总任务数: {n_tasks_to_process_total:,}")
print(f"起始任务ID: {start_task_id}")
print(f"每批任务数: {N_TASKS_PER_EXPERIMENT}")
print(f"并行核心数: {N_PARALLEL_JOBS}")
print(f"预计批次数: {n_tasks_to_process_total // N_TASKS_PER_EXPERIMENT}")
print(f"预计总时间: 4-6 小时")
print(f"{'='*80}\n")


while n_tasks_to_process_total > 0:
    # 计算本批任务数
    n_tasks_to_process_in_experiment = min(N_TASKS_PER_EXPERIMENT, n_tasks_to_process_total)

    print(
        f"【批次 {n_job+1}】启动任务 {start_task_id} - {start_task_id + n_tasks_to_process_in_experiment - 1}"
        f" (剩余 {n_tasks_to_process_total:,} 条)"
    )

    exp_name = f"{config.env_id}-{config.robot_id}-joint_joint-one-{config.planner}"

    launcher = Launcher(
        exp_name=exp_name,
        exp_file="generate_trajectories",
        project_name="rizon10s_million",
        start_seed=start_task_id,
        n_seeds=1,
        n_exps_in_parallel=N_EXPS_IN_PARALLEL,
        n_cores=N_CORES,
        memory_per_core=MEMORY_PER_CORE,
        days=0,
        hours=23,
        minutes=59,
        seconds=0,
        partition=PARTITION,
        conda_env=CONDA_ENV,
        gres=GRES,
        use_timestamp=False,
        check_results_directories=False,
        compact_dirs=True,
        base_dir=str(PROJECT_ROOT / "data" / "rizon"),
    )

    launcher.add_experiment(
        env_id__=config.env_id,
        robot_id__=config.robot_id,
        cfg_file="None",
        sample_joint_position_goals_with_same_ee_pose__=config.sample_joint_position_goals_with_same_ee_pose,
        selection__="one",
        planner__=config.planner,
        start_task_id=start_task_id,
        num_tasks=n_tasks_to_process_in_experiment,
        num_trajectories_per_task=config.num_trajectories_per_task,
        min_distance_robot_env=config.min_distance_robot_env,
        simplify_path=True,
        planner_allowed_time=config.planner_allowed_time,
        fit_bspline=False,
        bspline_num_control_points=config.bspline_num_control_points,
        bspline_degree=config.bspline_degree,
        bspline_zero_vel_at_start_and_goal=True,
        bspline_zero_acc_at_start_and_goal=True,
        n_parallel_jobs=N_PARALLEL_JOBS,  # 使用全部32核并行
        device="cpu",
        debug=False,
    )

    launcher.run(LOCAL, TEST)

    # 更新计数器
    start_task_id += n_tasks_to_process_in_experiment
    n_tasks_to_process_total -= n_tasks_to_process_in_experiment
    n_job += 1

    # 批次间短暂休息
    time.sleep(2)

print(f"\n{'='*80}")
print(f"所有批次已启动！")
print(f"总批次数: {n_job}")
print(f"数据将保存到: data/rizon/{exp_name}/")
print(f"\n监控进度:")
print(f"  ls -lh data/rizon/{exp_name}/")
print(f"{'='*80}\n")
