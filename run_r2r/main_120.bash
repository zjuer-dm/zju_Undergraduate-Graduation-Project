export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

# 120角度版本 - 单卡测试脚本

flag1="--exp_name release_r2r_dagger_120_test
      --run-type dagger
      --exp-config vlnce_baselines_120/config/r2r_baselines/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 2
      IL.iters 30000
      IL.lr 1e-5
      IL.log_every 200
      IL.ml_weight 1.0
      IL.sample_ratio 0.75
      IL.decay_interval 2000
      IL.warmup_iters 500
      IL.min_lr_ratio 1.0
      IL.load_from_ckpt False
      IL.is_requeue False
      IL.waypoint_aug True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      TASK_CONFIG.DATASET.SUFFIX _90
      MODEL.pretrained_path pretrained/r2r_rxr_ce_4cam/mlm.sap_habitat_depth/ckpts/model_step_455000.pt
      "

mode=$1
case $mode in 
      dagger)
      echo "###### dagger train mode (120 angles, single GPU) ######"
      python -m torch.distributed.launch --nproc_per_node=1 --master_port $2 run_120.py $flag1
      ;;
esac

# 单卡测试命令：
# CUDA_VISIBLE_DEVICES=0 bash run_r2r/main_120.bash dagger 9999
