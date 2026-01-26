# Repository Guidelines

## Project Structure & Module Organization
- `mpd/` is the core Python package (models, datasets, parametric trajectories, training loop, inference, and `torch_robotics`).
- `scripts/` contains runnable entry points for data generation, training, and inference (see `scripts/generate_data/`, `scripts/train/`, `scripts/inference/`).
- Configuration lives in `data_generation_cfgs/` (dataset generation) and `scripts/inference/cfgs/` (inference).
- `deps/` vendors third-party code/submodules; Isaac Gym is expected under `deps/isaacgym` alongside `pybullet_ompl` and `theseus`.
- Assets and outputs live in `figures/`, `logs/`, and downloaded datasets are typically symlinked into `data_trajectories/` and `data_trained_models/`.

## Build, Test, and Development Commands
- Install: `bash setup.sh` (or `bash setup_fixed.sh` if the fixed installer is needed), then `source set_env_variables.sh` and `conda activate mpd-splines-public`.
- Inference: `python scripts/inference/inference.py` (update `cfg_inference_path` or `scripts/inference/cfgs/*.yaml` for different environments).
- Training: `python scripts/train/train.py`; batch runs use `python scripts/train/launch_train_*.py`.
- Data generation: `python scripts/generate_data/generate_trajectories.py`; merge with `python scripts/generate_data/post_process_generated_dataset.py --data_dir ./data/env-robot/`; visualize with `python scripts/generate_data/visualize_trajectories.py --data_dir ./data/env-robot/`.

## Coding Style & Naming Conventions
- Python uses 4-space indentation, snake_case for files and functions, and CapWords for classes.
- Formatting: Black (line length 120) and isort (profile black) are configured in `pyproject.toml`. Pre-commit hooks are available in `.pre-commit-config.yaml`.

## Testing Guidelines
- There is no first-party test suite under `mpd/`; validate changes by running the relevant script path (data generation, training, or inference).
- If you modify vendored `deps/theseus`, run its tests from that directory: `python -m pytest tests`.

## Commit & Pull Request Guidelines
- Commit history favors short, imperative, sentence-case messages with a trailing period (e.g., "Fix installation.").
- There is no PR template; include a concise summary, commands run, and any config/data changes. Note external requirements (for example, Isaac Gym download) when relevant.
