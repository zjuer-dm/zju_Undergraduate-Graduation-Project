# ETP-R1 Copilot Instructions

This repository implements **ETP-R1: Evolving Topological Planning with Reinforcement Fine-tuning for Vision-Language Navigation in Continuous Environments**. It is built on top of `habitat-lab` and `habitat-sim`.

## Project Architecture

### Core Components
- **`vlnce_baselines/`**: Contains the main training logic, models, and RL algorithms.
  - **Trainers**: `GRPO_trainer_ETP_R1.py` (GRPO-R1), `ss_trainer_ETP_R1.py`.
  - **Models**: `models/R1Policy.py`, `models/etp/`, `models/vlnbert/`.
  - **Common**: `common/env_utils.py` (environment construction), `common/aux_losses.py`.
- **`habitat_extensions/`**: Custom extensions to the Habitat simulator.
  - **Config**: `config/default.py` extends the base Habitat config.
  - **Sensors**: `sensors.py` (e.g., `GlobalGPSSensor`, `RxRInstructionSensor`).
  - **Actions**: `task.py` (e.g., `MoveHighToLowAction`).
- **`run_r2r/` & `run_rxr/`**: Scripts and configs for running experiments on R2R and RxR datasets.
- **`precompute_img_features/`**: Scripts for extracting image features (RGB/Depth) using ResNet/CLIP.
- **`waypoint-predictor/`**: Standalone module for predicting waypoints.

### Data Flow
1.  **Input**: Matterport3D scenes (`data/scene_datasets/mp3d`) and navigation instructions.
2.  **Simulation**: `habitat-sim` renders observations.
3.  **Features**: Precomputed features or real-time extraction.
4.  **Policy**: `R1Policy` (Graph-based + VLN-BERT) processes observations and instructions.
5.  **Action**: High-level waypoints are converted to low-level actions via `habitat_extensions`.

## Critical Workflows

### Environment Setup
- **Conda Environment**: `etpr1`
- **Habitat-Sim**: Headless version required (`habitat-sim=0.1.7` headless).
- **Habitat-Lab**: Installed in develop mode (`v0.1.7`).

### Training & Evaluation
- **Entry Point**: `run.py` is the main entry point, typically invoked via bash scripts.
- **Launch Scripts**: Use `run_r2r/main_server.bash` or `run_rxr/main_server.bash`.
  - These scripts handle distributed training setup (`torch.distributed.launch`).
  - **Key Flags**:
    - `SIMULATOR_GPU_IDS`, `TORCH_GPU_IDS`: GPU allocation.
    - `NUM_ENVIRONMENTS`: Number of environments per GPU.
    - `exp-config`: Path to the YAML config file.
    - `run-type`: `dagger` or `grpo`.

### Configuration Management
- **System**: Uses `yacs` (via `habitat.config`).
- **Base Config**: `habitat_extensions/config/default.py`.
- **Experiment Config**: YAML files in `run_*/` (e.g., `run_r2r/iter_train.yaml`).
- **Overrides**: Command-line arguments in bash scripts override YAML configs.
  - Example: `TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True`

## Coding Conventions & Patterns

### Habitat Integration
- **Registry**: Use `@baseline_registry.register_trainer` to register new trainers.
- **Extensions**: Register custom sensors and measurements in `habitat_extensions/__init__.py` or their respective files.
- **Vector Environments**: `VectorEnv` is used to manage multiple simulator instances in parallel.

### Distributed Training
- The codebase assumes Distributed Data Parallel (DDP).
- **Rank Handling**: Use `local_rank` from `config` to handle process-specific logic (e.g., saving checkpoints only on rank 0).

### Path Handling
- **Absolute Paths**: Scripts often require absolute paths for datasets and features.
- **Connectivity Graphs**: Located in `precompute_img_features/connectivity/`.

## Common Tasks

### Adding a New Sensor
1.  Define the sensor class in `habitat_extensions/sensors.py`.
2.  Register it with `@registry.register_sensor`.
3.  Add default config in `habitat_extensions/config/default.py`.
4.  Enable it in the experiment YAML file.

### Modifying the Model
- **Policy**: Edit `vlnce_baselines/models/R1Policy.py`.
- **Graph Logic**: Check `vlnce_baselines/models/graph_utils.py` and `etp/`.
- **Inputs**: Ensure `net` method in policy handles the observation space defined in the config.
