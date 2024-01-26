EXPERIMENT=ornaments_001_baseline
GROUP=Omint3D
NAME=ornaments_001_baseline
CONFIG=projects/neuralangelo/configs/custom/ornaments_001.yaml
GPUS=2 # use >1 for multi-GPU training!
torchrun --nproc_per_node=${GPUS} train.py \
    --logdir=logs/${GROUP}/${NAME} \
    --config=${CONFIG} \
    --show_pbar
    # --checkpoint=logs/Omint3D/ornaments_001_baseline/epoch_03400_iteration_000170000_checkpoint.pt
