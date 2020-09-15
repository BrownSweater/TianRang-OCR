# export NCCL_P2P_DISABLE=1
export NGPUS=6
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 -m torch.distributed.launch --master_port=$((RANDOM + 10000)) --nproc_per_node=$NGPUS tools/train.py --config_file "config/ccpd_cvat_det_dbnet_shufflenet.yaml"