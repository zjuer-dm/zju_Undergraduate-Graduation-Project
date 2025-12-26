# ETP-R1 项目修改记录

## 2025-12-26: 12目 → 4目相机配置迁移

### 概述
将原有12目相机（12×3=36视角）配置修改为4目相机（4×3=12视角）配置，以适配实际硬件。

---

### 1. 预计算特征提取 (`precompute_img_features/`)

| 文件 | 修改内容 |
|------|----------|
| `save_img.py` | `VIEWPOINT_SIZE`: 36→12, `VFOV`: 60→90, heading增量 1.0→3.0, 仰角切换 %12→%4, `sys.path` 更新 |
| `extract_rgb_features.py` | `VIEWPOINT_SIZE`: 36→12, `batch_size`: 36→12, img_db路径 vfov60→vfov90 |
| `extract_depth_features.py` | `VIEWPOINT_SIZE`: 36→12, `batch_size`: 36→12, img_db路径 vfov60→vfov90 |
| `run.bash` | `PYTHONPATH` 更新为 `/home/wdm/Matterport3DSimulator_copy/build` |

---

### 2. Waypoint Predictor (`waypoint-predictor/`)

| 文件 | 修改内容 |
|------|----------|
| `run_waypoint.bash` | `ANGLES`: 40, `NUM_IMGS`: 4, `BATCH_SIZE`: 32, `CUDA_VISIBLE_DEVICES=0` |
| `image_encoders.py` | ddppo路径更新为 `/home/wdm/ICRA2026_etpnav/data/ddppo-models/` |
| `gen_training_data/get_images_inputs.py` | `NUMBER`: 4, 路径更新 |
| `gen_training_data/get_nav_dict.py` | `NUMBER`: 40, 路径更新 |
| `gen_training_data/test_twm0.2_obstacle_first.py` | `ANGLES`: 40 |

**数据生成完成**: `training_data/rgbd_fov90_4cam/`, `40_*_mp3d_waypoint_*.json`

---

### 3. 预训练代码 (`pretrain_src/pretrain_src_4/`)

| 文件 | 修改内容 |
|------|----------|
| `data/common.py` | `get_view_rel_angles()`: 12 views, 90°增量, 每4视角切换仰角 |
| `data/dataset.py` | `all_point_rel_angles`: range(12), 视角索引 12→4, heading %12→%4, 30°→90°, nav_types 36→12 |
| `run_pt_4/run_mix_server.bash` | 脚本路径更新为 `pretrain_src_4` 和 `run_pt_4`, 输出目录 `r2r_rxr_ce_4cam` |

---

### 4. SFT 训练代码 (`vlnce_baselines/`)

| 文件 | 修改内容 |
|------|----------|
| `waypoint_pred/TRM_net.py` | `num_angles`: 120→40, `num_imgs`: 12→4 |
| `waypoint_pred/utils.py` | `get_attention_mask` 默认参数: 12→4 |
| `utils.py` | 新增 `get_camera_orientations4()` 函数 (90°间隔) |
| `common/utils.py` | 新增 `get_camera_orientations4()` 函数 (90°间隔) |
| `common/base_il_trainer.py` | 导入并使用 `get_camera_orientations4` |
| `models/R1Policy.py` | `pano_img_idxes`: 0-11→0-3, `NUM_ANGLES`: 120→40, `NUM_IMGS`: 12→4 |
| `ss_trainer_ETP_R1.py` | 导入 `get_camera_orientations4`, `cand_idxes`: 12→4, `nav_types`: 12→4 |
| `GRPO_trainer_ETP_R1.py` | 同上 SFT trainer 的修改 |

---

### 关键参数映射

| 参数 | 12目 (原) | 4目 (新) |
|------|-----------|----------|
| 总视角数 | 36 | 12 |
| 每层相机数 | 12 | 4 |
| 水平FOV | 30° | 90° |
| VFOV | 60° | 90° |
| Waypoint ANGLES | 120 | 40 |
| NUM_IMGS (SFT) | 12 | 4 |

---

### 待完成
- [ ] 生成4目预训练特征文件 (CLIP, Depth)
- [ ] 配置 `mix_pretrain_server.json` 中的特征文件路径
- [ ] 训练 Waypoint Predictor
- [ ] 运行预训练
- [ ] 运行 SFT 训练
- [ ] 运行 GRPO 训练
