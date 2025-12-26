# Waypoint Predictor 4-Camera Configuration Guide

This document summarizes all modifications made to convert the waypoint predictor from 12-camera to 4-camera configuration.

## Configuration Overview

| Configuration | 12-Camera (Original) | 4-Camera (New) |
|--------------|---------------------|----------------|
| Horizontal cameras | 12 | 4 |
| Elevation levels | 3 (up/mid/down) | 3 (up/mid/down) |
| Total viewpoints | 36 (12×3) | 12 (4×3) |
| Horizontal FOV gap | 30° | 90° |
| ANGLES parameter | 120 | 40 |
| NUM_IMGS parameter | 12 | 4 |

## Modified Files Summary

### 1. Data Generation Scripts

#### `gen_training_data/get_images_inputs.py`
- `NUMBER`: 12 → **4**
- `scene_path`: Updated to `/home/wdm/vln-ce/data/...`
- `image_path`: `./training_data/rgbd_fov90/` → `./training_data/rgbd_fov90_4cam/`
- `RAW_GRAPH_PATH`: Updated to `/home/wdm/habitat_connectivity_graph/%s.json`

#### `gen_training_data/get_nav_dict.py`
- `NUMBER`: 120 → **40**
- `scene_path`: Updated to `/home/wdm/vln-ce/data/...`
- `RAW_GRAPH_PATH`: Updated to `/home/wdm/habitat_connectivity_graph/%s.json`

#### `gen_training_data/test_twm0.2_obstacle_first.py`
- `ANGLES`: 120 → **40**
- `RAW_GRAPH_PATH`: Updated to `/home/wdm/habitat_connectivity_graph/%s.json`

### 2. Training Scripts

#### `run_waypoint.bash`
- `EXP_ID`: `wp-train` → `wp-train-4cam`
- `ANGLES`: 120 → **40**
- `NUM_IMGS`: 12 → **4**

#### `waypoint_predictor.py`
- `train_img_dir`: Updated to `./training_data/rgbd_fov90_4cam/train/*/*.pkl`
- `eval_img_dir`: Updated to `./training_data/rgbd_fov90_4cam/val_unseen/*/*.pkl`

#### `image_encoders.py`
- Depth encoder weights path: Updated to `/home/wdm/vln-ce/data/ddppo-models/gibson-2plus-resnet50.pth`

### 3. New Scripts

#### `gen_training_data/run_gen_data.bash`
One-click script to generate all training data for both `train` and `val_unseen` splits.

## Required Data Dependencies

Before running the scripts, ensure these paths are correct on your Ubuntu server:

| Data Type | Path | Description |
|-----------|------|-------------|
| MP3D Scenes | `/home/wdm/vln-ce/data/scene_datasets/mp3d/{scan}/{scan}.glb` | Matterport3D scene data |
| Connectivity Graphs | `/home/wdm/habitat_connectivity_graph/{split}.json` | train.json, val_unseen.json |
| DDPPO Weights | `/home/wdm/vln-ce/data/ddppo-models/gibson-2plus-resnet50.pth` | Pre-trained depth encoder |

> **Note**: If your paths are different, search for `TODO: Change` comments in the modified files.

## Usage Instructions

### Step 1: Generate Training Data

```bash
cd waypoint-predictor
bash gen_training_data/run_gen_data.bash
```

This will generate:
- `training_data/rgbd_fov90_4cam/train/` - RGBD images for training
- `training_data/rgbd_fov90_4cam/val_unseen/` - RGBD images for validation
- `gen_training_data/nav_dicts/navigability_dict_train.json`
- `gen_training_data/nav_dicts/navigability_dict_val_unseen.json`
- `training_data/40_train_mp3d_waypoint_twm0.2_obstacle_first_withpos.json`
- `training_data/40_val_unseen_mp3d_waypoint_twm0.2_obstacle_first_withpos.json`

### Step 2: Train Waypoint Predictor

```bash
bash run_waypoint.bash
```

The trained model will be saved to `checkpoints/wp-train-4cam/`.

### Step 3: Evaluate (Optional)

Modify `run_waypoint.bash`:
```bash
--TRAINEVAL eval
```

## Output Files Structure

```
waypoint-predictor/
├── training_data/
│   ├── rgbd_fov90_4cam/
│   │   ├── train/{scan}/{scan}_{node}_mp3d_imgs.pkl
│   │   └── val_unseen/{scan}/{scan}_{node}_mp3d_imgs.pkl
│   ├── 40_train_mp3d_waypoint_twm0.2_obstacle_first_withpos.json
│   └── 40_val_unseen_mp3d_waypoint_twm0.2_obstacle_first_withpos.json
├── gen_training_data/
│   └── nav_dicts/
│       ├── navigability_dict_train.json
│       └── navigability_dict_val_unseen.json
└── checkpoints/
    └── wp-train-4cam/
        └── snap/
            ├── check_val_best_avg_wayscore
            ├── check_val_best_avg_pred_distance
            └── check_latest
```

## Technical Notes

- The ANGLES parameter (40 for 4-cam) represents the fine angular resolution for waypoint prediction
- The ratio is preserved: 120 ANGLES / 12 cameras = 10 angles per camera = 40 ANGLES / 4 cameras
- NUM_IMGS directly corresponds to the number of horizontal camera views (4)
- Distance discretization remains unchanged at 12 bins (0.25m intervals, max 3.25m)
