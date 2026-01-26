# Motion Planning Diffusion: 使用扩散模型学习和适配机器人运动规划

[![paper](https://img.shields.io/badge/Paper-%F0%9F%93%96-lightgray)](https://ieeexplore.ieee.org/abstract/document/11097366)
[![arXiv](https://img.shields.io/badge/arXiv-2502.08378-brown)](https://arxiv.org/abs/2412.19948)
[![](https://img.shields.io/badge/Website-%F0%9F%9A%80-yellow)](https://sites.google.com/view/motionplanningdiffusion/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)]()


<div style="display: flex; text-align:center; justify-content: center">
    <img src="figures/EnvSimple2D-RobotPointMass2D-joint_joint-one-RRTConnect.gif" alt="Image 1" width="400" style="display: inline-block;">
    <img src="figures/EnvWarehouse-RobotPanda-config_file_v01-joint_joint-one-RRTConnect.gif" alt="Image 2" width="400" style="display: inline-block;">
</div>

本仓库实现了 Motion Planning Diffusion (**MPD**) —— 一种利用扩散模型进行学习和规划机器人运动的方法。

该项目的一个旧版本已经弃用，但仍可在 [https://github.com/jacarvalho/mpd-public](https://github.com/jacarvalho/mpd-public) 获取。

如果您有任何问题，请联系我 —— [joao@robot-learning.de](mailto:joao@robot-learning.de)

---
# 安装

前置条件：
- Ubuntu 22.04（可能也适用于更高版本）
- [miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html)

克隆此仓库：
```bash
mkdir -p ~/Projects/MotionPlanningDiffusion/
cd ~/Projects/MotionPlanningDiffusion/
git clone --recurse-submodules git@github.com:joaoamcarvalho/mpd-splines-public.git mpd-splines-public
cd mpd-splines-public
```

下载 [IsaacGym Preview 4](https://developer.nvidia.com/isaac-gym) 并将其解压到 `deps/isaacgym` 目录下：
```bash
mv ~/Downloads/IsaacGym_Preview_4_Package.tar.gz ~/Projects/MotionPlanningDiffusion/mpd-splines-public/deps/
cd ~/Projects/MotionPlanningDiffusion/mpd-splines-public/deps
tar -xvf IsaacGym_Preview_4_Package.tar.gz
```

运行 bash 设置脚本以安装所有内容（这可能需要一段时间）：
```bash
bash setup.sh
```

在运行任何脚本之前，请务必设置环境变量并激活 conda 环境：
```bash
source set_env_variables.sh
conda activate mpd-splines-public
```

---
## 下载数据集和预训练模型

下载链接：https://drive.google.com/file/d/1KG5ejn0g0KkDuUK6tPUqfmRYCNoKzK4K/view?usp=drive_link

```bash
tar -xvf data_public.tar.gz
ln -s data_public/data_trajectories data_trajectories
ln -s data_public/data_trained_models data_trained_models
```


---
## 使用预训练模型进行推理

[scripts/inference/cfgs](scripts/inference/cfgs) 下的配置文件包含了推理的超参数。\
在 `scripts/inference/inference.py` 文件中，您可以更改 `cfg_inference_path` 参数来尝试针对不同环境训练的模型。

```bash
cd scripts/inference
python inference.py
python inference.py --planner_alg rrtconnect_then_guide --n_trajectory_samples 1

```
对比

```bash
source /home/bochu/code/mpd/mpd-splines-public/set_env_variables.sh && conda activate mpd-splines-public 

python inference.py > mpd_results.log 2>&1 && python inference.py --planner_alg rrtconnect_then_guide --n_trajectory_samples 1 > hybrid_results.log 2>&1
```

---
# 训练先验模型（从头开始）


## 数据生成

生成数据需要很长时间，因此我们建议[下载数据集](#下载数据集和预训练模型)。
但如果您仍然想生成自己的数据，可以使用 `scripts/generate_data` 文件夹中的脚本。

前往 `scripts/generate_data` 文件夹。

基础脚本是：
```bash
python generate_trajectories.py
```

要并行生成多个数据集，请修改 `launch_generate_trajectories.py` 脚本：
```bash
python launch_generate_trajectories.py
```

生成数据后，运行后处理文件将所有数据合并到一个 hdf5 文件中。
然后，您可以通过翻转轨迹路径来使数据集翻倍。
```bash
python post_process_trajectories.py --help
python flip_solution_paths.py  (修改 PATH_TO_DATASETS 变量)
```

要可视化生成的数据，使用 `visualize_trajectories.py` 脚本：
```bash
python visualize_trajectories.py
```

---
## 训练模型

训练脚本位于 `scripts/train` 文件夹中。

基础脚本是：
```bash
cd scripts/train
python train.py
```

要并行训练多个模型，请使用 `launch_train_*` 文件。


---
## 引用

如果您使用了我们的工作或代码库，请引用我们的文章：
```latex
@article{carvalho2025motion,
  title={Motion planning diffusion: Learning and adapting robot motion planning with diffusion models},
  author={Carvalho, Jo{\~a}o and Le, An T and Kicki, Piotr and Koert, Dorothea and Peters, Jan},
  journal={IEEE Transactions on Robotics},
  year={2025},
  publisher={IEEE}
}

@inproceedings{carvalho2023motion,
  title={Motion planning diffusion: Learning and planning of robot motions with diffusion models},
  author={Carvalho, Jo{\~a}o and Le, An T and Baierl, Mark and Koert, Dorothea and Peters, Jan},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2023}
}
```


---
## 鸣谢

本工作及软件的部分内容取自或启发自：
- [https://github.com/jannerm/diffuser](https://github.com/jannerm/diffuser)
