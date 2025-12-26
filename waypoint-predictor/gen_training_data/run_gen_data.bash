#!/bin/bash
# Script to generate all waypoint predictor training data (4-camera version)
# Run this script from the waypoint-predictor directory
#
# Prerequisites:
#   1. MP3D scene datasets at: /home/wdm/ETPNav/data/scene_datasets/mp3d/
#   2. Habitat connectivity graphs at: habitat_connectivity_graph_folder/
#   3. Python environment with habitat-sim installed
#
# Usage:
#   cd waypoint-predictor
#   bash gen_training_data/run_gen_data.bash

set -e  # Exit on error

echo "=========================================="
echo "Waypoint Predictor Training Data Generation"
echo "4-Camera Configuration (4 views)"
echo "=========================================="

# Check Python environment
echo ""
echo "[Step 0] Checking Python environment..."
python --version

# Create necessary directories
echo ""
echo "[Step 1] Creating output directories..."
mkdir -p training_data/rgbd_fov90_4cam/train
mkdir -p training_data/rgbd_fov90_4cam/val_unseen
mkdir -p gen_training_data/nav_dicts
mkdir -p training_data

# ==========================================
# Step 2: Generate RGBD images for train split
# ==========================================
echo ""
echo "=========================================="
echo "[Step 2] Generating RGBD images for 'train' split..."
echo "Output: training_data/rgbd_fov90_4cam/train/"
echo "=========================================="

# Set SPLIT to train
sed -i "s/SPLIT = 'val_unseen'/SPLIT = 'train'/g" gen_training_data/get_images_inputs.py
sed -i "s/SPLIT = 'val_seen'/SPLIT = 'train'/g" gen_training_data/get_images_inputs.py
python gen_training_data/get_images_inputs.py
echo "[Step 2] Train RGBD images generation completed!"

# ==========================================
# Step 3: Generate RGBD images for val_unseen split
# ==========================================
echo ""
echo "=========================================="
echo "[Step 3] Generating RGBD images for 'val_unseen' split..."
echo "Output: training_data/rgbd_fov90_4cam/val_unseen/"
echo "=========================================="

# Set SPLIT to val_unseen
sed -i "s/SPLIT = 'train'/SPLIT = 'val_unseen'/g" gen_training_data/get_images_inputs.py
python gen_training_data/get_images_inputs.py
echo "[Step 3] Val_unseen RGBD images generation completed!"

# ==========================================
# Step 4: Generate navigability dict for train split
# ==========================================
echo ""
echo "=========================================="
echo "[Step 4] Generating navigability dict for 'train' split..."
echo "Output: gen_training_data/nav_dicts/navigability_dict_train.json"
echo "=========================================="

# Set SPLIT to train
sed -i "s/SPLIT = 'val_unseen'/SPLIT = 'train'/g" gen_training_data/get_nav_dict.py
sed -i "s/SPLIT = 'val_seen'/SPLIT = 'train'/g" gen_training_data/get_nav_dict.py
python gen_training_data/get_nav_dict.py
echo "[Step 4] Train navigability dict generation completed!"

# ==========================================
# Step 5: Generate navigability dict for val_unseen split
# ==========================================
echo ""
echo "=========================================="
echo "[Step 5] Generating navigability dict for 'val_unseen' split..."
echo "Output: gen_training_data/nav_dicts/navigability_dict_val_unseen.json"
echo "=========================================="

# Set SPLIT to val_unseen
sed -i "s/SPLIT = 'train'/SPLIT = 'val_unseen'/g" gen_training_data/get_nav_dict.py
python gen_training_data/get_nav_dict.py
echo "[Step 5] Val_unseen navigability dict generation completed!"

# ==========================================
# Step 6: Generate final training data with obstacle info
# ==========================================
echo ""
echo "=========================================="
echo "[Step 6] Generating final training data with obstacle info..."
echo "Output: training_data/40_train_mp3d_waypoint_twm0.2_obstacle_first_withpos.json"
echo "        training_data/40_val_unseen_mp3d_waypoint_twm0.2_obstacle_first_withpos.json"
echo "=========================================="
python gen_training_data/test_twm0.2_obstacle_first.py
echo "[Step 6] Final training data generation completed!"

echo ""
echo "=========================================="
echo "All data generation completed successfully!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - training_data/rgbd_fov90_4cam/train/          (RGBD images)"
echo "  - training_data/rgbd_fov90_4cam/val_unseen/     (RGBD images)"
echo "  - gen_training_data/nav_dicts/navigability_dict_train.json"
echo "  - gen_training_data/nav_dicts/navigability_dict_val_unseen.json"
echo "  - training_data/40_train_mp3d_waypoint_twm0.2_obstacle_first_withpos.json"
echo "  - training_data/40_val_unseen_mp3d_waypoint_twm0.2_obstacle_first_withpos.json"
echo ""
echo "Next step: Run 'bash run_waypoint.bash' to train the waypoint predictor"
