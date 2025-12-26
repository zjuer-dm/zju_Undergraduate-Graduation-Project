#!/bin/bash
# Script to generate all waypoint predictor training data (4-camera version)
# Run this script from the waypoint-predictor directory
#
# Prerequisites:
#   1. MP3D scene datasets at: /home/wdm/vln-ce/data/scene_datasets/mp3d/
#   2. Habitat connectivity graphs at: /home/wdm/habitat_connectivity_graph/
#   3. Python environment with habitat-sim installed
#
# Usage:
#   cd waypoint-predictor
#   bash gen_training_data/run_gen_data.bash

set -e  # Exit on error

echo "=========================================="
echo "Waypoint Predictor Training Data Generation"
echo "4-Camera Configuration (4 views * 3 elevations = 12 total)"
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

# Create a temporary modified version of the script for 'train' split
cat > gen_training_data/get_images_inputs_train.py << 'EOF'
import json
import numpy as np
import utils
import habitat
import os
import pickle
from habitat.sims import make_sim


config_path = './gen_training_data/config.yaml'
# TODO: Change to your data path
scene_path = '/home/wdm/vln-ce/data/scene_datasets/mp3d/{scan}/{scan}.glb'
image_path = './training_data/rgbd_fov90_4cam/'
save_path = os.path.join(image_path,'{split}/{scan}/{scan}_{node}_mp3d_imgs.pkl')
# TODO: Change to your connectivity graph path
RAW_GRAPH_PATH= '/home/wdm/habitat_connectivity_graph/%s.json'
# 4-camera setup: 4 horizontal views (was 12 for 12-camera)
NUMBER = 4

SPLIT = 'train'

with open(RAW_GRAPH_PATH%SPLIT, 'r') as f:
    raw_graph_data = json.load(f)

nav_dict = {}
total_invalids = 0
total = 0

for scene, data in raw_graph_data.items():
    ''' connectivity dictionary '''
    connect_dict = {}
    for edge_id, edge_info in data['edges'].items():
        node_a = edge_info['nodes'][0]
        node_b = edge_info['nodes'][1]

        if node_a not in connect_dict:
            connect_dict[node_a] = [node_b]
        else:
            connect_dict[node_a].append(node_b)
        if node_b not in connect_dict:
            connect_dict[node_b] = [node_a]
        else:
            connect_dict[node_b].append(node_a)

    '''make sim for obstacle checking'''
    config = habitat.get_config(config_path)
    config.defrost()
    config.TASK.SENSORS = []
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.25
    config.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = False
    config.SIMULATOR.SCENE = scene_path.format(scan=scene)
    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)

    '''save images'''
    if not os.path.exists(image_path+'{split}/{scan}'.format(split=SPLIT,scan=scene)):
        os.makedirs(image_path+'{split}/{scan}'.format(split=SPLIT,scan=scene))
    navigability_dict = {}
    
    i = 0
    total = len(connect_dict)
    for node_a, neighbors in connect_dict.items():
        navigability_dict[node_a] = utils.init_single_node_dict(number=NUMBER)
        rgbs = []
        depths = []
        node_a_pos = np.array(data['nodes'][node_a])[[0, 2]]

        habitat_pos = np.array(data['nodes'][node_a])
        for info in navigability_dict[node_a].values():
            position, heading = habitat_pos, info['heading']
            theta = -(heading - np.pi) / 2
            rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
            obs = sim.get_observations_at(position, rotation)
            rgbs.append(obs['rgb'])
            depths.append(obs['depth'])
        with open(save_path.format(split=SPLIT, scan=scene, node=node_a), 'wb') as f:
            pickle.dump({'rgb': np.array(rgbs),
                         'depth': np.array(depths, dtype=np.float16)}, f)
        utils.print_progress(i+1,total)
        i+=1

    sim.close()
EOF

python gen_training_data/get_images_inputs_train.py
echo "[Step 2] Train RGBD images generation completed!"

# ==========================================
# Step 3: Generate RGBD images for val_unseen split
# ==========================================
echo ""
echo "=========================================="
echo "[Step 3] Generating RGBD images for 'val_unseen' split..."
echo "Output: training_data/rgbd_fov90_4cam/val_unseen/"
echo "=========================================="

# Create a temporary modified version for 'val_unseen' split
sed 's/SPLIT = .train./SPLIT = '\''val_unseen'\''/' gen_training_data/get_images_inputs_train.py > gen_training_data/get_images_inputs_val.py
python gen_training_data/get_images_inputs_val.py
echo "[Step 3] Val_unseen RGBD images generation completed!"

# ==========================================
# Step 4: Generate navigability dict for train split
# ==========================================
echo ""
echo "=========================================="
echo "[Step 4] Generating navigability dict for 'train' split..."
echo "Output: gen_training_data/nav_dicts/navigability_dict_train.json"
echo "=========================================="

# Create a temporary modified version for 'train' split
cat > gen_training_data/get_nav_dict_train.py << 'EOF'
import json
import numpy as np
import utils
import habitat
from habitat.sims import make_sim
from utils import Simulator

config_path = 'gen_training_data/config.yaml'
# TODO: Change to your data path
scene_path = '/home/wdm/vln-ce/data/scene_datasets/mp3d/{scan}/{scan}.glb'
# TODO: Change to your connectivity graph path
RAW_GRAPH_PATH= '/home/wdm/habitat_connectivity_graph/%s.json'
# 4-camera setup: 40 angle bins (was 120 for 12-camera, ratio 120/12=10, so 4*10=40)
NUMBER = 40

SPLIT = 'train'

with open(RAW_GRAPH_PATH%SPLIT, 'r') as f:
    raw_graph_data = json.load(f)

nav_dict = {}
total_invalids = 0
total = 0

for scene, data in raw_graph_data.items():
    ''' connectivity dictionary '''
    connect_dict = {}
    for edge_id, edge_info in data['edges'].items():
        node_a = edge_info['nodes'][0]
        node_b = edge_info['nodes'][1]

        if node_a not in connect_dict:
            connect_dict[node_a] = [node_b]
        else:
            connect_dict[node_a].append(node_b)
        if node_b not in connect_dict:
            connect_dict[node_b] = [node_a]
        else:
            connect_dict[node_b].append(node_a)


    '''make sim for obstacle checking'''
    config = habitat.get_config(config_path)
    config.defrost()
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.25
    config.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = False
    config.SIMULATOR.TYPE = 'Sim-v1'
    config.SIMULATOR.SCENE = scene_path.format(scan=scene)
    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)

    ''' process each node to standard data format '''
    navigability_dict = {}
    total = len(connect_dict)
    for i, pair in enumerate(connect_dict.items()):
        node_a, neighbors = pair
        navigability_dict[node_a] = utils.init_single_node_dict(number=NUMBER)
        node_a_pos = np.array(data['nodes'][node_a])[[0,2]]
    
        habitat_pos = np.array(data['nodes'][node_a])
        for id, info in navigability_dict[node_a].items():
            obstacle_distance, obstacle_index = utils.get_obstacle_info(habitat_pos,info['heading'],sim)
            info['obstacle_distance'] = obstacle_distance
            info['obstacle_index'] = obstacle_index
    
        for node_b in neighbors:
            node_b_pos = np.array(data['nodes'][node_b])[[0,2]]
    
            edge_vec = (node_b_pos - node_a_pos)
            angle, angleIndex, distance, distanceIndex = utils.edge_vec_to_indexes(edge_vec,number=NUMBER)
    
            navigability_dict[node_a][str(angleIndex)]['has_waypoint'] = True
            navigability_dict[node_a][str(angleIndex)]['waypoint'].append(
                {
                    'node_id': node_b,
                    'position': node_b_pos.tolist(),
                    'angle': angle,
                    'angleIndex': angleIndex,
                    'distance': distance,
                    'distanceIndex': distanceIndex,
                })
        utils.print_progress(i+1,total)
    
    nav_dict[scene] = navigability_dict
    sim.close()

output_path = './gen_training_data/nav_dicts/navigability_dict_%s.json'%SPLIT
with open(output_path, 'w') as fo:
    json.dump(nav_dict, fo, ensure_ascii=False, indent=4)
EOF

python gen_training_data/get_nav_dict_train.py
echo "[Step 4] Train navigability dict generation completed!"

# ==========================================
# Step 5: Generate navigability dict for val_unseen split
# ==========================================
echo ""
echo "=========================================="
echo "[Step 5] Generating navigability dict for 'val_unseen' split..."
echo "Output: gen_training_data/nav_dicts/navigability_dict_val_unseen.json"
echo "=========================================="

sed 's/SPLIT = .train./SPLIT = '\''val_unseen'\''/' gen_training_data/get_nav_dict_train.py > gen_training_data/get_nav_dict_val.py
python gen_training_data/get_nav_dict_val.py
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

# ==========================================
# Cleanup temporary files
# ==========================================
echo ""
echo "[Cleanup] Removing temporary files..."
rm -f gen_training_data/get_images_inputs_train.py
rm -f gen_training_data/get_images_inputs_val.py
rm -f gen_training_data/get_nav_dict_train.py
rm -f gen_training_data/get_nav_dict_val.py

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
