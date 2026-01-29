# MPD Splines Public - 项目深度文档

## 1. 项目概览
**Motion Planning Diffusion (MPD)** 是一个基于扩散模型（Diffusion Models）的机器人运动规划框架。本项目 (`mpd-splines-public`) 是 MPD 的官方实现，特别侧重于使用样条曲线（B-Splines）作为轨迹的参数化表示。

核心思想是将运动规划问题重构为生成式建模问题：训练一个扩散模型，从高斯噪声中去噪生成连接起点和终点的无碰撞、平滑轨迹。利用可微运动学和规划目标函数，生成的轨迹可以进一步优化。

## 2. 目录结构与关键文件详解

```
mpd-splines-public/
├── mpd/                        # 核心 Python 包
│   ├── models/                 # 神经网络模型定义
│   │   ├── diffusion_models/   # 扩散模型核心实现 (DDPM, DDIM, 采样逻辑)
│   │   │   ├── diffusion_model_base.py # 高斯扩散过程，DDPM/DDIM 采样循环
│   │   │   ├── models.py       # TemporalUnet 网络架构定义
│   ├── torch_robotics/         # 可微机器人学库 (核心组件)
│   │   ├── environments/       # 仿真环境 (EnvWarehouse, EnvTableShelf 等)
│   │   ├── robots/             # 机器人定义 (RobotPanda, RobotPointMass 等)
│   │   ├── torch_kinematics_tree/ # 可微正运动学实现
│   │   └── visualizers/        # 可视化工具
│   ├── parametric_trajectory/  # 轨迹参数化表示
│   │   ├── trajectory_bspline.py   # B-Spline 参数化 (主要使用)
│   │   └── trajectory_waypoints.py # 传统路径点参数化
│   ├── trainer/                # 训练循环逻辑
│   ├── datasets/               # 数据加载器 (HDF5 数据集读取)
│   ├── inference/              # 推理与评估脚本
│   └── ...
├── scripts/                    # 可执行脚本
│   ├── generate_data/          # 轨迹数据生成 (基于 OMPL)
│   │   └── generate_trajectories.py # RRTConnect 数据生成主脚本
│   ├── train/                  # 模型训练脚本
│   └── inference/              # 评估与推理脚本
├── deps/                       # 外部依赖
│   ├── isaacgym/               # NVIDIA Isaac Gym (物理仿真)
│   ├── pybullet_ompl/          # OMPL Python 绑定 (用于生成专家数据)
│   └── ...
├── data_generation_cfgs/       # 数据生成的配置文件
├── setup_fixed.sh              # 推荐的安装脚本
├── environment.yml             # Conda 环境定义
└── README.md                   # 官方说明
```

## 3. 核心技术模块

### 3.1 扩散模型 (`mpd.models.diffusion_models`)
本项目使用的是基于分数的去噪扩散概率模型 (DDPM, Denoising Diffusion Probabilistic Models)。

*   **架构 (`models.py`)**: 使用了一个 `TemporalUnet`。这是一个处理时序数据的一维 U-Net，包含：
    *   **ResidualTemporalBlock**: 处理时间序列特征的残差块。
    *   **Self-Attention / Linear Attention**: 用于捕捉长距离的时间依赖关系。
    *   **SpatialTransformer**: 用于处理条件注入（Conditioning），如将起点、终点或环境信息注入到网络中。
    *   **Downsample1d / Upsample1d**: 在时间维度上进行下采样和上采样，构建 U-Net 的层级结构。
*   **扩散过程 (`diffusion_model_base.py`)**:
    *   **前向过程**: 使用预定义的噪声调度（默认是 cosine 或 exponential schedule）向轨迹添加高斯噪声。
    *   **反向过程**: 网络预测噪声（`predict_epsilon=True`）或直接预测去噪后的轨迹（`predict_epsilon=False`）。
    *   **采样算法**: 支持标准的 DDPM 采样和加速的 DDIM (Denoising Diffusion Implicit Models) 采样。
    *   **引导 (Guidance)**: 在采样过程中支持基于梯度的引导，可以利用 `torch_robotics` 中的可微损失函数（如碰撞损失）来修正生成过程。

### 3.2 可微机器人学 (`mpd.torch_robotics`)
这是一个独立的库，允许在 PyTorch 中进行机器人相关的计算，并支持自动微分。

*   **Robot (`robots/`)**:
    *   `RobotPanda`: Franka Emika Panda 机械臂模型。
    *   `RobotPlanarLink`: 平面 2 连杆或 4 连杆机械臂。
    *   `RobotPointMass`: 2D 点质量模型（用于简单测试）。
*   **Environment (`environments/`)**: 定义了障碍物和边界。
    *   `EnvWarehouse`: 仓库环境，包含货架等障碍物。
    *   `EnvTableShelf`: 桌子和架子环境。
    *   `EnvSpheres3D`: 充满球体障碍物的 3D 空间。
    *   这些环境支持基于 SDF (Signed Distance Function) 的碰撞检测，这是完全可微的。

### 3.3 轨迹参数化 (`mpd.parametric_trajectory`)
MPD 的一个关键特性是它不直接预测离散的路径点，而是预测 B-Spline 的控制点。
*   **优势**: B-Spline 天然保证了轨迹的平滑性（C2 连续），并且大大降低了需要预测的参数数量（即控制点数量远少于时间步数量）。
*   **实现**: `trajectory_bspline.py` 实现了从控制点到密集轨迹点的转换矩阵运算。

## 4. 详细工作流

### 4.1 安装
推荐使用 `setup_fixed.sh` 进行一键安装。
```bash
bash setup.sh
```
该脚本完成以下关键步骤：
1.  创建名为 `mpd-splines-public` 的 Conda 环境。
2.  安装 PyTorch 和其他 Python 依赖。
3.  **编译 OMPL**: 在 `deps/pybullet_ompl` 下编译 OMPL 的 Python 绑定。这是一个耗时步骤。
4.  **注意**: Isaac Gym 需要你手动下载并放置在 `deps/isaacgym` 目录下，脚本不会自动下载。

### 4.2 第一步：专家数据生成
训练扩散模型需要大量的“专家演示”数据。
*   **脚本**: `scripts/generate_data/generate_trajectories.py`
*   **原理**:
    1.  使用 `pybullet` 加载机器人和环境的 URDF。
    2.  调用 OMPL (Open Motion Planning Library) 中的算法（默认是 `RRTConnect`）。
    3.  生成无碰撞的路径，并可选地进行路径简化和平滑（B-Spline拟合）。
    4.  数据被保存为 HDF5 格式，存储在 `data_trajectories/` 中。
*   **配置**: 可以在 `experiment` 装饰器中修改 `env_id`, `robot_id`, `num_tasks` 等参数。

### 4.3 第二步：模型训练
*   **脚本**: `scripts/train/train.py`
*   **流程**:
    1.  加载 HDF5 数据集。
    2.  初始化 `GaussianDiffusionModel` 和 `TemporalUnet`。
    3.  训练目标是最小化预测噪声与添加噪声之间的 L2 损失（MSE）。
    4.  模型会学习从纯噪声中恢复出符合环境约束的运动轨迹分布。
*   **输出**: 训练日志和模型权重保存在 `results/` 目录下。

### 4.4 第三步：推理与评估
*   **脚本**: `scripts/inference/inference.py`
*   **流程**:
    1.  加载训练好的模型权重。
    2.  给定新的起点和终点（作为 Condition）。
    3.  模型从高斯噪声开始，经过多次去噪迭代（Diffusion Steps），生成建议轨迹。
    4.  **后处理/细化 (Refinement)**: 生成的轨迹可以使用 `torch_robotics` 的梯度进行微调，进一步优化平滑度或避障性能。

## 5. 关键配置参数说明
在训练和生成数据时，以下参数至关重要：

*   `diffusion_steps`: 扩散过程的步数，通常为 100 或 1000。步数越多生成质量越高，但推理速度越慢。
*   `variance_schedule`: 噪声调度策略，`cosine` 通常比 `linear` 效果更好。
*   `n_support_points`: 轨迹的控制点数量。如果使用 B-Spline，这对应于控制点的数量。
*   `predict_epsilon`: 是否预测噪声。设为 `True` 通常训练更稳定。

## 6. 常见问题
*   **OMPL 编译失败**: 确保系统安装了必要的构建工具 (`cmake`, `make`, `gcc`)。如果是 SSH 权限问题，请参考之前的会话切换到 HTTPS。
*   **Isaac Gym 缺失**: 这是一个闭源库，必须从 NVIDIA官网下载并解压到指定位置。

## 7. 训练指标与 Loss 详解

在 WandB 或训练日志中，您会看到一系列关键指标。以下详解结合了代码实现逻辑 (`mpd/models/diffusion_models/` 和 `mpd/torch_robotics/tasks/`)。

### 7.1 损失函数 (Loss Functions)

所有的 Loss 本质上都是衡量模型**“去噪能力”**的指标。

*   **代码位置**: `mpd/models/diffusion_models/helpers.py` (Class `WeightedL2`)
*   **核心逻辑**: `diffusion_model_base.py` -> `p_losses` 函数

*   **计算公式 (均方误差 MSE)**:
    $$ L = \frac{1}{N} \sum (\mathbf{\epsilon} - \mathbf{\epsilon}_\theta(\mathbf{x}_t, t))^2 $$
    我们在比较两个**三维张量**（`Batch_Size` x `Control_Points` x `Joints`）之间的差异。
    *   $\mathbf{\epsilon}$ (**Targets/真实值**): 也就是代码中的 `noise`。我们在训练时人为添加到轨迹中的**高斯噪声 (Ground Truth Noise)**。
    *   $\mathbf{\epsilon}_\theta$ (**Predictions/预测值**): 也就是代码中的 `x_recon` (当 `predict_epsilon=True` 时)。模型接收带噪轨迹后，预测出的**噪声分布**。
    *   **直观理解**: 就像给一段清晰的录音加上电流声（加噪），然后让模型把电流声分离出来（预测噪声）。分离得越干净，Loss 越低。

*   **三个 Loss 的区别**:
    1.  **`diffusion_loss`**: **“每道练习题的扣分”**。实时监控当前训练 Batch 的去噪误差。
    2.  **`total_loss`**: **“本次作业的总扣分”**。在单任务训练中，它完全等同于 `diffusion_loss`。
    3.  **`VALIDATION total_loss`** (至关重要): **“模拟考试的扣分”**。
        *   它使用模型**从未见过**的验证集数据进行测试。
        *   **如何判断拟合**:
            *   若 `Train Loss` 低且 `Val Loss` 也低 -> **真学霸 (完美拟合)**。
            *   若 `Train Loss` 低但 `Val Loss` 高 -> **死记硬背 (过拟合)**。

### 7.2 生成质量指标 (Validation Metrics)

这些指标衡量生成的轨迹是否真正可用。
**注意：计算这些指标时，模型会一次性生成约 250 条轨迹（25条/任务 × 10个任务）进行统计。**

*   **代码位置**: `mpd/summaries/summary_trajectory_generation.py` -> `summary_fn`
*   **核心实现**: `mpd/torch_robotics/torch_robotics/tasks/tasks.py` (Class `PlanningTask`)

#### 1. `VALIDATION percentage free trajs` (无碰撞比例)
*   **别名**: **“考试不撞车率”**
*   **代码对应**: `PlanningTask.compute_fraction_valid_trajs`
*   **计算逻辑**: **一票否决制**。
    *   代码调用 `get_trajs_unvalid_and_valid`，检查轨迹上的每一个点（包括插值点）。
    *   判定标准：
        1.  **碰撞检测**: `torch.logical_not(trajs_waypoints_collisions).all(dim=-1)` —— 所有时间步都不允许碰撞。
        2.  **关节限位**: `(position >= q_min) & (position <= q_max)` —— 所有关节角度必须在物理限制内。
    *   **公式**: 
        $$ \text{FreeRate} = \frac{\text{Count}(\text{Valid Trajectories})}{\text{Total Trajectories} (250)} $$
*   **趋势**: 通常会剧烈震荡（0.2 - 0.8），这很正常，反映了验证集中不同任务的难度差异。

#### 2. `VALIDATION percentage collision intensity` (碰撞强度)
*   **别名**: **“事故严重程度”**
*   **代码对应**: `PlanningTask.compute_collision_intensity_trajs`
*   **计算逻辑**: **按点计费**。
    *   它不关心整条轨迹是否作废，只关心有多少个时间点发生了碰撞。
    *   **公式**:
        $$ \text{Intensity} = \frac{\text{Count}(\text{Collision Points})}{\text{Count}(\text{All Points in All Trajectories})} $$
*   **意义**: 即使 `Free Trajs` 很低，如果这个值也很低（如 0.05），说明大部分轨迹只是轻微擦边，很容易修复。

#### 3. `VALIDATION success` (成功率)
*   **别名**: **“任务完成率” (兜底能力)**
*   **代码对应**: `PlanningTask.compute_success_valid_trajs`
*   **计算逻辑**: **只要有一条能用就行**。
    *   代码逻辑：`if trajs_valid.nelement() >= 1: return 1`
    *   只要从生成的 250 条候选轨迹里挑出**至少 1 条**满足上述 `Free Trajs` 标准的轨迹，该任务就被判定为成功。
*   **公式**: 这是一个二值指标 (0 或 1)。WandB 上显示的是多个验证 Batch 的平均值。
*   **意义**: 如果达到 1.0，说明模型非常可靠，无论遇到什么任务，总能生成可行的解决方案。这是部署模型时最重要的信心指标。
