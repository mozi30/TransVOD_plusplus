#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`
source /root/activate msda
EXP_DIR=/root/TemporalAttentionPlayground/TransVOD_plusplus/exps/swinbase_ref_frame_multi_interval
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}

ref_frames=(1 3 7 15)

for r in "${ref_frames[@]}"; do
    python -u main.py \
        --backbone swin_b_p4w7 \
        --epochs 10 \
        --num_feature_levels 1 \
        --lr 2e-6 \
        --num_queries 100 \
        --hidden_dim 256 \
        --dilation \
        --batch_size 1 \
        --num_workers 1 \
        --num_ref_frames $r \
        --resume /root/weights/transvod/checkpoint_ref_single_frame_new.pth \
        --with_box_refine \
        --vid_path '/root/datasets/visdrone/transvod' \
        --dataset_file 'vid_multi' \
        --output_dir ${EXP_DIR}${r}/ \
        --interval1 8 \
        ${PY_ARGS} 2>&1 | tee ${EXP_DIR}${r}/log.train.$T
done