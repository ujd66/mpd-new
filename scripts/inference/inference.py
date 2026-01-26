import sys
import fractions
import math

# 猴子补丁 (Monkey Patch)：用于兼容性修复
# 修复 networkx 2.2 在 Python 3.9+ 中使用已弃用的 fractions.gcd 的问题
if not hasattr(fractions, "gcd"):
    fractions.gcd = math.gcd

from mpd.utils.patches import numpy_monkey_patch

# 应用 numpy 的补丁，解决版本兼容性问题
numpy_monkey_patch()

import time
from functools import partial

# import isaacgym


from dotmap import DotMap

import gc
import os
from pprint import pprint

import numpy as np
import torch

# Monkey Patch: 修改 torch.load 的默认行为
# 将 default weights_only 设置为 False，以允许加载旧版 pickle 文件
# 这里的 lambda 函数保留了原有参数，并强制覆盖 weights_only 参数
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(*args, **{**kwargs, "weights_only": False})
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from experiment_launcher import single_experiment_yaml, run_experiment
from mpd.inference.inference import EvaluationSamplesGenerator, GenerativeOptimizationPlanner, render_results
from mpd.metrics.metrics import PlanningMetricsCalculator
from mpd.utils.loaders import get_planning_task_and_dataset, load_params_from_yaml, save_to_yaml

# from torch_robotics.isaac_gym_envs.motion_planning_envs import (
#     MotionPlanningIsaacGymEnv,
#     MotionPlanningControllerIsaacGym,
# )
from torch_robotics.robots import RobotPanda
from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_torch, to_numpy

allow_ops_in_compiled_graph()


@single_experiment_yaml
def experiment(
    ########################################################################
    # 模型和推理参数配置路径
    # 可以在这里取消注释选择不同的配置文件，分别对应不同的环境和机器人任务
    # cfg_inference_path: str = './cfgs/config_EnvNarrowPassageDense2D-RobotPointMass2D_00.yaml',
    # cfg_inference_path: str = './cfgs/config_EnvPlanar2Link-RobotPlanar2Link_00.yaml',
    # cfg_inference_path: str = './cfgs/config_EnvPlanar4Link-RobotPlanar4Link_00.yaml',
    # cfg_inference_path: str = "./cfgs/config_EnvSimple2D-RobotPointMass2D_00.yaml",
    cfg_inference_path: str = "./cfgs/config_EnvSpheres3D-RobotPanda_00.yaml",
    # cfg_inference_path: str = "./cfgs/config_EnvWarehouse-RobotPanda-config_file_v01_00.yaml",
    ########################################################################
    # 数据集选择：使用训练集还是验证/测试集进行评估
    selection_start_goal: str = "validation",  # training, validation/test
    ########################################################################
    # 评估的起始和目标状态数量
    n_start_goal_states: int = 30,
    ########################################################################
    # 是否以低内存模式保存单个规划结果
    save_results_single_plan_low_mem: bool = False,
    ########################################################################
    # 可视化选项
    render_joint_space_time_iters: bool = True,  # 渲染关节空间随时间的变化
    render_joint_space_env_iters: bool = False,  # 渲染关节空间环境迭代
    render_env_robot_opt_iters: bool = False,  # 渲染机器人优化迭代过程
    render_env_robot_trajectories: bool = False,  # 渲染机器人轨迹
    render_pybullet: bool = True,  # 是否在 PyBullet 中进行可视化
    draw_collision_spheres: bool = False,  # 是否绘制碰撞球
    run_evaluation_issac_gym: bool = False,  # 是否运行 Isaac Gym 评估（需要Isaac Gym环境）
    render_isaacgym_viewer: bool = False,  # 是否显示 Isaac Gym 查看器
    render_isaacgym_movie: bool = False,  # 是否录制 Isaac Gym 视频
    ########################################################################
    # 设备配置 (CPU 或 CUDA)
    device: str = "cuda:0",  # cpu, cuda
    debug: bool = False,  # 是否开启调试模式
    ########################################################################
    # 必须参数
    # seed: int = int(time.time()), # 随机种子
    seed: int = 1,
    results_dir: str = "logs",  # 结果保存目录
    ########################################################################
    **kwargs,
):
    # 设置随机种子以保证可复现性
    fix_random_seed(seed)

    # 获取 Torch 设备对象
    device = get_torch_device(device)
    tensor_args = {"device": device, "dtype": torch.float32}

    # 保存和加载推理配置
    # 从 yaml 文件加载配置参数
    args_inference = DotMap(load_params_from_yaml(cfg_inference_path))
    # 允许命令行参数覆盖 YAML 配置 (Arguments from CLI overwrite YAML config)
    for k, v in kwargs.items():
        if k in args_inference:
            current_v = args_inference[k]
            # 尝试将命令行参数转换为配置中原有的类型 (Try to cast CLI arg to the type of config value)
            if current_v is not None and isinstance(v, str):
                try:
                    if isinstance(current_v, bool):
                        v = v.lower() in ("true", "t", "yes", "1")
                    else:
                        v = type(current_v)(v)
                except ValueError:
                    pass  # Keep original value if casting fails

            # 只有当值不同时才覆盖，避免无谓的操作，并打印提示
            if current_v != v:
                print(f"Overwriting config parameter '{k}': {current_v} -> {v} (from CLI)")
                args_inference[k] = v

    # 根据配置选择模型目录
    # 如果使用 CVAE (Conditional Variational Autoencoder)
    if "cvae" in args_inference.planner_alg:
        if args_inference.model_selection == "bspline":
            args_inference.model_dir = args_inference.model_dir_cvae_bspline
        elif args_inference.model_selection == "waypoints":
            args_inference.model_dir = args_inference.model_dir_cvae_waypoints
        else:
            raise NotImplementedError
    # 否则默认使用 DDPM (Denoising Diffusion Probabilistic Models)
    else:
        if args_inference.model_selection == "bspline":
            args_inference.model_dir = args_inference.model_dir_ddpm_bspline
        elif args_inference.model_selection == "waypoints":
            args_inference.model_dir = args_inference.model_dir_ddpm_waypoints
        else:
            raise NotImplementedError

    # 展开环境变量 (如 ${HOME})
    args_inference.model_dir = os.path.expandvars(args_inference.model_dir)

    # 将推理参数保存到 log 目录
    save_to_yaml(args_inference.toDict(), os.path.join(results_dir, "args_inference.yaml"))

    print(f"\n-------------------------------------------------------------------------------------------------")
    print(f"cfg_inference_path:\n{cfg_inference_path}")
    print(f"Model:\n{args_inference.model_dir}")
    print(f"--------------------------------------------------------------------------------------------------")

    ################################################################################################################
    # 加载数据集、环境、机器人和规划任务
    # 使用模型目录下的 args.yaml 加载训练时的参数
    args_train = DotMap(load_params_from_yaml(os.path.join(args_inference.model_dir, "args.yaml")))

    # 覆盖训练参数：
    # - 强制开启 gripper (夹爪)
    # - 关闭 reload_data (避免重新加载原始数据)
    # - 设置结果目录
    # - 加载索引
    args_train.update(
        **args_inference,
        gripper=True,
        reload_data=False,
        results_dir=results_dir,
        load_indices=True,
        tensor_args=tensor_args,
    )
    # 获取规划任务和数据集子集
    planning_task, train_subset, _, val_subset, _ = get_planning_task_and_dataset(**args_train)

    ################################################################################################################
    # 评估样本生成器
    # 用于从数据集生成或采样起始和目标状态，以及进行基于采样的规划（如RRTConnect）
    evaluation_samples_generator = EvaluationSamplesGenerator(
        planning_task,
        train_subset,
        val_subset,
        selection_start_goal=selection_start_goal,
        planner="RRTConnect",
        tensor_args=tensor_args,
        debug=debug,
        render_pybullet=render_pybullet,
        **args_inference,
    )

    ################################################################################################################
    # 加载生成式优化规划器 (Generative Optimization Planner)
    # 这是核心规划模块，结合了扩散模型/CVAE 和基于采样的规划器
    generative_optimization_planner = GenerativeOptimizationPlanner(
        planning_task,
        train_subset.dataset,
        args_train,
        args_inference,
        tensor_args,
        # 这里的 sampling_based_planner_fn 是一个偏函数，用于在需要时调用 OMPL 进行规划
        sampling_based_planner_fn=partial(
            evaluation_samples_generator.generate_data_ompl_worker.run,
            planner_allowed_time=10.0,
            interpolate_num=args_inference.num_T_pts,
            simplify_path=True,
        ),
        debug=debug,
    )

    ################################################################################################################
    # # IsaacGym environment and motion planning controller
    # motion_planning_isaac_env = None
    # if run_evaluation_issac_gym:
    #     robot_asset_file = planning_task.robot.robot_urdf_file
    #     if draw_collision_spheres:
    #         robot_asset_file = planning_task.robot.robot_urdf_collision_spheres_file
    #     motion_planning_isaac_env = MotionPlanningIsaacGymEnv(
    #         planning_task.env,
    #         planning_task.robot,
    #         asset_root=get_robot_path().as_posix(),
    #         robot_asset_file=robot_asset_file.replace(get_robot_path().as_posix() + "/", ""),
    #         num_envs=args_inference.n_trajectory_samples,
    #         # all_robots_in_one_env=True if n_start_goal_states == 1 else False,
    #         all_robots_in_one_env=True,
    #         render_isaacgym_viewer=render_isaacgym_viewer,
    #         render_camera_global=render_isaacgym_movie,
    #         render_camera_global_append_to_recorder=render_isaacgym_movie,
    #         sync_viewer_with_real_time=False,
    #         show_viewer=render_isaacgym_viewer,
    #         camera_global_from_top=True if planning_task.env.dim == 2 else False,
    #         add_ground_plane=False,
    #         viewer_time_between_steps=torch.diff(planning_task.parametric_trajectory.get_timesteps()[:2]).item(),
    #         draw_goal_configuration=True if not train_subset.dataset.context_ee_goal_pose else False,
    #         draw_ee_pose_goal=True if train_subset.dataset.context_ee_goal_pose else False,
    #         color_robots=False,
    #         draw_contact_forces=False,
    #         draw_end_effector_frame=False,
    #         draw_end_effector_path=True,
    #     )

    #     motion_planning_controller_isaac_gym = MotionPlanningControllerIsaacGym(motion_planning_isaac_env)

    ################################################################################################################
    # 指标计算器
    # 用于计算路径长度、平滑度、碰撞情况等评估指标
    planning_metrics_calculator = PlanningMetricsCalculator(planning_task)

    ################################################################################################################
    # 顺序规划多个起始和目标状态

    # 随机选择要规划的样本索引
    if selection_start_goal == "training":
        idx_sample_l = np.random.choice(np.arange(len(train_subset)), n_start_goal_states)
    else:
        idx_sample_l = np.random.choice(np.arange(len(val_subset)), n_start_goal_states)

    # 遍历每个选定的样本进行规划
    for idx_sg, idx_sample in enumerate(idx_sample_l):
        print(f"\n-------------------------------------------------------------------------------------------------")
        print(f"----------------PLANNING {idx_sg+1}/{n_start_goal_states}------------------")
        print(f"--------------------------------------------------------------------------------------------------")

        results_single_plan = DotMap(t_generator=0.0, t_guide=0.0)

        # 获取当前样本的起始关节位置、目标关节位置和目标末端执行器位姿
        q_pos_start, q_pos_goal, ee_pose_goal = evaluation_samples_generator.get_data_sample(idx_sg)

        print("\n----------------START AND GOAL states----------------")
        print(f"q_pos_start: {q_pos_start}")
        print(f"q_pos_goal: {q_pos_goal}")
        print(f"ee_pose_goal: {ee_pose_goal}")

        if debug:
            evaluation_samples_generator.add_start_goal_marker(q_pos_start, q_pos_goal)

        ############################################################################################################
        # 运行运动规划推理
        # plan_trajectory 方法会调用底层的生成模型（如 Diffusion U-Net）生成轨迹，
        # 可能会包含引导（Guidance）和优化过程
        print(f"\n----------------PLAN TRAJECTORIES----------------")
        print(f"Starting inference...")
        results_single_plan = generative_optimization_planner.plan_trajectory(
            q_pos_start, q_pos_goal, ee_pose_goal, results_ns=results_single_plan, debug=debug
        )
        print(f"...inference finished.")

        ############################################################################################################
        # 在 PyBullet 中展示最佳轨迹
        if render_pybullet and results_single_plan.q_trajs_pos_best is not None:
            time.sleep(3)
            ########################
            # PyBullet 可视化
            q_pos_path = to_numpy(results_single_plan.q_trajs_pos_best)

            # 为 Panda 机器人添加夹爪状态 (如果是 Panda 且轨迹维度只有7维)
            # 机器人完整维度通常为9 (7关节 + 2夹爪)
            if (
                isinstance(planning_task.robot, RobotPanda)
                and q_pos_path.shape[1] == 7
                and evaluation_samples_generator.generate_data_ompl_worker.pbompl_interface.robot.num_dim == 9
            ):
                q_pos_path = np.concatenate((q_pos_path, np.zeros((q_pos_path.shape[0], 2))), axis=-1)

            # 从轨迹时长和时间点数量计算 dt (时间步长)
            # 这是为了确保可视化播放的速度与实际物理时间一致
            dt = planning_task.parametric_trajectory.trajectory_duration / planning_task.parametric_trajectory.num_T_pts
            evaluation_samples_generator.generate_data_ompl_worker.pbompl_interface.execute(q_pos_path, sleep_time=dt)

        ############################################################################################################
        # # Evaluate and show in IsaacGym
        # isaacgym_statistics = None
        # if run_evaluation_issac_gym and results_single_plan.q_trajs_pos_valid is not None:
        #     ########################
        #     motion_planning_isaac_env.ee_pose_goal = planning_task.robot.get_EE_pose(
        #         to_torch(q_pos_goal.unsqueeze(0), device), flatten_pos_quat=True, quat_xyzw=True
        #     ).squeeze(0)

        #     # Execute all valid trajectories
        #     if results_single_plan.q_trajs_pos_valid.shape[0] > 0:
        #         q_trajs_pos = results_single_plan.q_trajs_pos_valid.movedim(1, 0)  # horizon, batch, D
        #         isaacgym_statistics = motion_planning_controller_isaac_gym.execute_trajectories(
        #             q_trajs_pos,
        #             q_pos_starts=q_trajs_pos[0],
        #             q_pos_goal=q_trajs_pos[-1][0],  # add steps for better visualization
        #             n_pre_steps=5 if render_isaacgym_viewer or render_isaacgym_movie else 0,
        #             n_post_steps=5 if render_isaacgym_viewer or render_isaacgym_movie else 0,
        #             stop_robot_if_in_contact=False,
        #             make_video=render_isaacgym_movie,
        #             video_duration=args_inference.trajectory_duration,
        #             video_path=os.path.join(results_dir, f"isaacgym-{idx_sg:03d}.mp4"),
        #             make_gif=False,
        #         )
        #     results_single_plan.isaacgym_statistics = isaacgym_statistics

        ############################################################################################################
        # 计算运动规划指标
        print(f"\n----------------METRICS----------------")
        results_single_plan.metrics = planning_metrics_calculator.compute_metrics(results_single_plan)

        print(f"t_inference_total: {results_single_plan.t_inference_total:.3f} sec")
        print(f"t_generator: {results_single_plan.t_generator:.3f} sec")
        print(f"t_guide: {results_single_plan.t_guide:.3f} sec")

        print(f"isaacgym_statistics:")
        pprint(results_single_plan.isaacgym_statistics)

        print(f"metrics:")
        pprint(results_single_plan.metrics)

        # 保存数据
        # 如果开启低内存模式，只保存关键数据
        results_single_plan_to_save = results_single_plan
        if save_results_single_plan_low_mem:
            results_single_plan_to_save = DotMap(
                t_generator=results_single_plan.t_generator,
                t_guide=results_single_plan.t_guide,
                t_inference_total=results_single_plan.t_inference_total,
                q_pos_start=q_pos_start,
                q_pos_goal=q_pos_goal,
                ee_pose_goal=ee_pose_goal,
                control_points_iters=results_single_plan.control_points_iters,
                metrics=results_single_plan.metrics,
                isaacgym_statistics=results_single_plan.isaacgym_statistics,
            )

        # 使用 torch.save 保存结果到 .pt 文件
        torch.save(
            results_single_plan_to_save,
            os.path.join(results_dir, f"results_single_plan-{idx_sg:03d}.pt"),
            _use_new_zipfile_serialization=True,
        )

        # 显式清理内存，防止在多轮循环中显存溢出 (Explicitly clear memory to prevent OOM)
        gc.collect()
        torch.cuda.empty_cache()

        ############################################################################################################
        # 渲染采样结果
        # 这个函数会生成包含轨迹、优化过程等的可视化图表并保存
        render_results(
            args_inference,
            planning_task,
            q_pos_start,
            q_pos_goal,
            results_single_plan,
            idx_sg,
            results_dir,
            render_joint_space_time_iters=render_joint_space_time_iters,
            render_joint_space_env_iters=render_joint_space_env_iters,
            render_planning_env_robot_opt_iters=render_env_robot_opt_iters,
            render_planning_env_robot_trajectories=render_env_robot_trajectories,
            debug=debug,
        )

        ############################################################################################################
        # empty memory
        del results_single_plan
        gc.collect()
        torch.cuda.empty_cache()

    ################################################################################################################
    # 清理资源
    # 终止 OMPL 工作线程
    evaluation_samples_generator.generate_data_ompl_worker.terminate()
    # if motion_planning_isaac_env is not None:
    #     motion_planning_isaac_env.clean_up()
    #     del motion_planning_isaac_env
    #     del motion_planning_controller_isaac_gym

    # 强制进行垃圾回收和清空 CUDA 缓存
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run_experiment(experiment)
