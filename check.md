# Code Check Log

## [vlnce_baselines/common] vs [vlnce_baselines_orin/common]
- **Status: PASS**
- **Analysis**: Differences in `utils.py` correct (`get_camera_orientations4` vs `12`). `base_il_trainer.py` properly calls corresponding orientation functions. No logic errors found.

## [vlnce_baselines/models] vs [vlnce_baselines_orin/models]
- **Status: PASS**
- **Analysis**: `encoders` are identical. `R1Policy.py` adapted for 4 cameras (NUM_ANGLES=40, angles CCW) vs 12 cameras (NUM_ANGLES=120, angles CW). Logic is self-consistent for 4-camera setup.

##- [x] `vlnce_baselines/waypoint_pred/utils.py` vs `vlnce_baselines_orin/waypoint_pred/utils.py`: **Verified**.
  - **Difference**: `get_attention_mask` default `num_imgs` is 4 vs 12.
  - **Status**: Correct. Consistent with 4-camera setup.

## Summary of Findings
- **Critical Issue**: `run_r2r/iter_train.yaml` points to 12-camera pretraining weights (`mlm.sap_r2r`).
- **Minor Issue**: `NUM_ANGLES` in `iter_train.yaml` is unused but incorrect (12 vs 40).
- **Codebase Integrity**:
  - `TRM_net.py`: Correctly adapted (40 angles, RGB+Depth).
  - `R1Policy.py`: Correctly adapted (40 angles, 4 images, CCW logic).
  - `utils.py`: Correctly adapted (4 orientations).
  - `waypoint_pred/utils.py`: Correctly adapted (default num_imgs=4).
- **Conclusion**: The codebase is correctly adapted for 4 cameras. The poor performance is solely due to loading incompatible pretraining weights.

## Next Step
- Update `run_r2r/iter_train.yaml` to use the correct 4-camera pretraining weights.
