import os
from collections import OrderedDict
from itertools import product
from pathlib import Path

import yaml

from experiment_launcher import Launcher
from experiment_launcher.utils import is_local

# ----------------------------------------------------------------------------------------------------------------------
# 该脚本的作用：
# 这是一个批量实验启动器。它通过组合不同的参数（环境、算法、模型等），自动生成对应的配置文件，
# 并调用 `inference.py` 来执行具体的实验。
# 它支持在本地运行（串行）或提交到 SLURM 集群（并行）。
# ----------------------------------------------------------------------------------------------------------------------

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

########################################################################################################################
# LAUNCHER 配置
# 这里的设置决定了实验是运行在本地还是集群上，以及并行的程度。

LOCAL = is_local() # 自动检测是否在本地机器上运行
USE_CUDA = True

N_EXPS_IN_PARALLEL = 1 # 并行运行的实验数量

N_CORES = N_EXPS_IN_PARALLEL * 6
MEMORY_SINGLE_JOB = 22000
PARTITION = "gpu" if USE_CUDA else "amd3,amd2,amd"
GRES = "gpu:1" if USE_CUDA else None
CONSTRAINT = "rtx3090|a5000" if USE_CUDA else None  # 指定 GPU 类型
CONDA_ENV = "mpd-splines" # 指定运行实验的 Conda 环境


EXPERIMENT_NAME = Path(__file__).stem

# 初始化 Launcher
# 它管理实验的调度。如果在本地运行，它会直接启动进程；如果是集群，它会生成 sbatch 脚本。
launcher = Launcher(
    exp_name=EXPERIMENT_NAME,
    exp_file="inference", # 对应 inference.py
    n_seeds=1,
    n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    n_cores=N_CORES,
    memory_per_core=N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES,
    days=2,
    hours=23,
    minutes=59,
    seconds=0,
    partition=PARTITION,
    conda_env=CONDA_ENV,
    gres=GRES,
    constraint=CONSTRAINT,
    use_timestamp=True,
)

########################################################################################################################
# 实验配置基准 (Setup experiments base)
# (配置文件路径, 对应的 ExtraObjects 环境名称)
config_files_bspline_extraobjectsenv_l = [
    ("./cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml", "EnvSimple2DExtraObjectsV00"),
    ("./cfgs/config_EnvNarrowPassageDense2D-RobotPointMass2D_00.yaml", "EnvNarrowPassageDense2DExtraObjectsV00"),
    ("./cfgs/config_EnvPlanar2Link-RobotPlanar2Link_00.yaml", "EnvPlanar2LinkExtraObjectsV00"),
    ("./cfgs/config_EnvPlanar4Link-RobotPlanar4Link_00.yaml", "EnvPlanar4LinkExtraObjectsV00"),
    ("./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml", "EnvSpheres3DExtraObjectsV00"),
    ("./cfgs/config_EnvWarehouse-RobotPanda-config_file_v01_00.yaml", "EnvWarehouseExtraObjectsV00"),
]

# 是否在环境中添加额外的障碍物对象
extra_objects_l = [False, True]

phase_time_class_l = [
    "PhaseTimeLinear",
    # 'PhaseTimeSigmoid',
]

project_gradient_hierarchy_l = [
    False,
    # True,
]


trajectory_duration = 10.0
n_trajectory_samples = 100

n_start_goal_states = 100

# 默认的可视化选项
default_options = OrderedDict(
    save_results_single_plan_low_memory=True,
    render_joint_space_time_iters=False,
    render_joint_space_env_iters=False,
    render_env_robot_opt_iters=False,
    render_env_robot_trajectories=False,
    render_pybullet=False, # 是否使用 PyBullet 可视化
    draw_collision_spheres=False,
    run_evaluation_issac_gym=False,
    render_isaacgym_viewer=False,
    render_isaacgym_movie=False,
)

# 清理并创建临时配置文件夹
import shutil

shutil.rmtree("./cfgs/tmp", ignore_errors=True)
os.makedirs("./cfgs/tmp", exist_ok=True)

exp_id = 0

########################################################################################################################
# 非扩散模型实验循环 (NON-DIFFUSION)
# 这里运行基准对比算法 (Baseline)，例如 CVAE, GP (Gaussian Process)

planner_alg_l = [
    # prior only
    "cvae",
    # prior and guide
    "gp_prior_then_guide",
    "cvae_prior_then_guide",
]

# 遍历所有参数组合
for (
    config_files_bspline_extraobjectsenv,
    extra_objects,
    phase_time_class,
    planner_alg,
    project_gradient_hierarchy,
) in product(
    config_files_bspline_extraobjectsenv_l,
    extra_objects_l,
    phase_time_class_l,
    planner_alg_l,
    project_gradient_hierarchy_l,
):

    # 创建临时配置文件
    cfg_file = config_files_bspline_extraobjectsenv[0]
    extraobjects_env = config_files_bspline_extraobjectsenv[1]
    # 读取基础 yaml 配置
    with open(cfg_file, "r") as fp:
        cfg_base = yaml.safe_load(fp)

    # 更新配置路径 (主要用于集群环境路径修复)
    if not LOCAL:
        for model_dir in ["model_dir_ddpm_bspline", "model_dir_cvae_bspline", "model_dir_ddpm_waypoints"]:
            cfg_base[model_dir] = "/mnt/beegfs" + cfg_base[model_dir]

    with open(os.path.join(cfg_base["model_dir_ddpm_bspline"], "args.yaml"), "r") as fp:
        model_dir_ddpm_args = yaml.safe_load(fp)
        dataset_subdir = model_dir_ddpm_args["dataset_subdir"]

    if extra_objects:
        cfg_base["env_id_replace"] = extraobjects_env
    else:
        cfg_base["env_id_replace"] = None
    cfg_base["phase_time_class"] = phase_time_class
    cfg_base["planner_alg"] = planner_alg
    cfg_base["project_gradient_hierarchy"] = project_gradient_hierarchy
    cfg_base["trajectory_duration"] = trajectory_duration
    cfg_base["n_trajectory_samples"] = n_trajectory_samples

    # 保存新的临时参数文件
    cfg_file_stem = Path(cfg_file).stem
    cfg_file_path_tmp = os.path.join("./cfgs/tmp", f"{cfg_file_stem}-{exp_id:04d}.yaml")
    with open(cfg_file_path_tmp, "w") as fp:
        yaml.dump(cfg_base, fp)

    # 添加实验任务到 Launcher
    launcher.add_experiment(
        # 传递给 inference.py 的参数
        dataset_subdir__=dataset_subdir,
        selection_start_goal__="validation",
        extra_objects__=extra_objects,
        planner_alg__=planner_alg,
        phase_time_class__=phase_time_class,
        project_gradient_hierarchy__=project_gradient_hierarchy,
        trajectory_duration__=trajectory_duration,
        cfg_inference_path=cfg_file_path_tmp,
        n_start_goal_states=n_start_goal_states,
        device="cuda:0",
        debug=False,
        # Visualization options
        **default_options,
    )

    exp_id += 1


########################################################################################################################
# 扩散模型实验循环 (DIFFUSION)
# 运行 MPD (Motion Planning Diffusion) 及其相关变体

model_selection_l = [
    "bspline",
    "waypoints",
]

planner_alg_l = [
    # prior only
    "diffusion_prior",
    # prior and guide
    "diffusion_prior_then_guide",
    "mpd",
]

diffusion_sampling_method_l = [
    # 'ddpm',
    "ddim"
]

n_diffusion_steps_without_noise_l = [
    0,
    # 1
]

for (
    config_files_bspline_extraobjectsenv,
    extra_objects,
    model_selection,
    phase_time_class,
    planner_alg,
    diffusion_sampling_method,
    n_diffusion_steps_without_noise,
    project_gradient_hierarchy,
) in product(
    config_files_bspline_extraobjectsenv_l,
    extra_objects_l,
    model_selection_l,
    phase_time_class_l,
    planner_alg_l,
    diffusion_sampling_method_l,
    n_diffusion_steps_without_noise_l,
    project_gradient_hierarchy_l,
):

    # 跳过无效组合
    if planner_alg != "mpd" and model_selection == "waypoints":
        # skip waypoints for planners that are not mpd
        continue

    # 创建临时配置文件
    cfg_file = config_files_bspline_extraobjectsenv[0]
    extraobjects_env = config_files_bspline_extraobjectsenv[1]
    
    with open(cfg_file, "r") as fp:
        cfg_base = yaml.safe_load(fp)

    # update the config file
    if not LOCAL:
        for model_dir in ["model_dir_ddpm_bspline", "model_dir_cvae_bspline", "model_dir_ddpm_waypoints"]:
            cfg_base[model_dir] = "/mnt/beegfs" + cfg_base[model_dir]

    with open(os.path.join(cfg_base["model_dir_ddpm_bspline"], "args.yaml"), "r") as fp:
        model_dir_ddpm_args = yaml.safe_load(fp)
        dataset_subdir = model_dir_ddpm_args["dataset_subdir"]

    if extra_objects:
        cfg_base["env_id_replace"] = extraobjects_env
    else:
        cfg_base["env_id_replace"] = None

    cfg_base["model_selection"] = model_selection

    cfg_base["phase_time_class"] = phase_time_class
    cfg_base["planner_alg"] = planner_alg
    cfg_base["diffusion_sampling_method"] = diffusion_sampling_method
    cfg_base["n_diffusion_steps_without_noise"] = n_diffusion_steps_without_noise
    cfg_base["project_gradient_hierarchy"] = project_gradient_hierarchy
    cfg_base["trajectory_duration"] = trajectory_duration
    cfg_base["n_trajectory_samples"] = n_trajectory_samples

    # save the new config file
    cfg_file_stem = Path(cfg_file).stem
    cfg_file_path_tmp = os.path.join("./cfgs/tmp", f"{cfg_file_stem}-{exp_id:04d}.yaml")
    with open(cfg_file_path_tmp, "w") as fp:
        yaml.dump(cfg_base, fp)

    # 添加实验任务
    launcher.add_experiment(
        # exp_id__=exp_id,
        # dummy variables for subfolders
        dataset_subdir__=dataset_subdir,
        selection_start_goal__="validation",
        extra_objects__=extra_objects,
        planner_alg__=planner_alg,
        model_selection__=model_selection,
        phase_time_class__=phase_time_class,
        diffusion_sampling_method__=diffusion_sampling_method,
        n_diffusion_steps_without_noise__=n_diffusion_steps_without_noise,
        project_gradient_hierarchy__=project_gradient_hierarchy,
        trajectory_duration__=trajectory_duration,
        cfg_inference_path=cfg_file_path_tmp,
        n_start_goal_states=n_start_goal_states,
        device="cuda:0",
        debug=False,
        # Visualization options
        **default_options,
    )

    exp_id += 1

# 运行所有添加的实验任务
launcher.run(LOCAL, test=False)
