#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`
WEIGHT_DIR=/root/weights/transvod
EXP_DIR=exps/transvod/intervall-perturbations/
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}


perturbations2=(
  gaussian_noise
  motion_blur
  jpeg_compression
  brightness_change
  contrast_change
  pixelation
  defocus_blur
)

severities=(low med high)

for p in "${perturbations2[@]}"; do
    for s in "${severities[@]}"; do
        echo "Evaluating with perturbation: $p, severity: $s"
        python -u main.py \
            --eval \
            --backbone swin_b_p4w7 \
            --num_feature_levels 1 \
            --num_queries 100 \
            --hidden_dim 256 \
            --dilation \
            --batch_size 16 \
            --resume ${WEIGHT_DIR}/checkpoint0001.pth \
            --num_workers 1 \
            --with_box_refine \
            --dataset_file 'vid_single' \
            --vid_path '/root/datasets/visdrone/transvod' \
            --output_dir ${EXP_DIR} \
            --perturbation \
            --select_perturbation $p \
            --severity $s \
            ${PY_ARGS} 2>&1 | tee -a ${EXP_DIR}/log.eval_e6.$T
    done
done

# python -u main.py \
#     --eval \
#     --backbone swin_b_p4w7 \
#     --epochs 7 \
#     --lr_drop_epochs 5 6 \
#     --num_feature_levels 1\
#     --num_queries 100 \
#     --dilation \
#     --batch_size 64 \
#     --hidden_dim 256 \
#     --num_workers 1 \
#     --with_box_refine \
#     --vid_path '/root/datasets/visdrone/transvod' \
#     --resume /root/TemporalAttentionPlayground/TransVOD_plusplus/exps/msda_swinbase_full_dataset/checkpoint0020.pth \
#     --dataset_file 'vid_single' \
#     --output_dir 'exps/msda_swinbase'
