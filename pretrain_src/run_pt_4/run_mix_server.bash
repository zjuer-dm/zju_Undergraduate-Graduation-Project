
NODE_RANK=0
NUM_GPUS=2
outdir=pretrained/r2r_rxr_ce_4cam/mlm.sap_habitat_depth

python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK --master_port=$1 \
    pretrain_src/pretrain_src_4/train_r2r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config pretrain_src/run_pt_4/mix_model_config_dep.json \
    --config pretrain_src/run_pt_4/mix_pretrain_server.json \
    --output_dir $outdir
# export NCCL_P2P_DISABLE=1
# CUDA_VISIBLE_DEVICES=6,7 bash pretrain_src/run_pt_4/run_mix_server.bash 2333
