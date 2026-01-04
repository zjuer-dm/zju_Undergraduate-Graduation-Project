#!/bin/bash
# 120角度训练数据生成脚本 (RGBD图像复用已有的)
# 使用: cd waypoint-predictor && bash gen_training_data/run_gen_data120.bash

set -e

echo "=== 生成120角度Waypoint训练数据 ==="

# Step 1: 生成train split的navigability dict
echo "[1/3] 生成 train navigability dict..."
sed -i "s/SPLIT = 'val_unseen'/SPLIT = 'train'/g" gen_training_data/get_nav_dict.py
python gen_training_data/get_nav_dict.py

# Step 2: 生成val_unseen split的navigability dict
echo "[2/3] 生成 val_unseen navigability dict..."
sed -i "s/SPLIT = 'train'/SPLIT = 'val_unseen'/g" gen_training_data/get_nav_dict.py
python gen_training_data/get_nav_dict.py

# Step 3: 生成最终训练数据
echo "[3/3] 生成最终训练数据 (with obstacle)..."
python gen_training_data/test_twm0.2_obstacle_first.py

echo ""
echo "=== 完成! ==="
echo "生成文件:"
echo "  - training_data/120_train_mp3d_waypoint_twm0.2_obstacle_first_withpos.json"
echo "  - training_data/120_val_unseen_mp3d_waypoint_twm0.2_obstacle_first_withpos.json"
echo ""
echo "下一步: bash run_waypoint.bash"
