
export CUDA_VISIBLE_DEVICES=0

flag="--EXP_ID wp-train-4cam

      --TRAINEVAL train
      --VIS 0

      --ANGLES 40
      --NUM_IMGS 4

      --EPOCH 300
      --BATCH_SIZE 32
      --LEARNING_RATE 1e-6

      --WEIGHT 0

      --TRM_LAYER 2
      --TRM_NEIGHBOR 1
      --HEATMAP_OFFSET 5
      --HIDDEN_DIM 768"

python waypoint_predictor.py $flag
