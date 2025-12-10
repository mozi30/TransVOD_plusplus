#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=/root/TemporalAttentionPlayground/TransVOD_plusplus/exps/test_random_ref_frames_multi
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
# python -u main.py \
#     --eval \
#     --backbone swin_b_p4w7 \
#     --epochs 6 \
#     --num_feature_levels 1 \
#     --num_queries 100 \
#     --hidden_dim 256 \
#     --dilation \
#     --batch_size 1 \
#     --num_ref_frames 1 \
#     --vid_path '/root/datasets/visdrone/transvod' \
#     --resume /root/TemporalAttentionPlayground/TransVOD_plusplus/exps/msda_swinbase_full_dataset_multi/checkpoint_ref_frame1.pth \
#     --lr_drop_epochs 4 5 \
#     --num_workers 1 \
#     --with_box_refine \
#     --dataset_file 'vid_multi' \
#     --output_dir ${EXP_DIR} \
#     ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.eval_e6.$T

python -u main.py \
    --eval \
    --backbone swin_b_p4w7 \
    --epochs 6 \
    --num_feature_levels 1 \
    --num_queries 100 \
    --hidden_dim 256 \
    --dilation \
    --batch_size 1 \
    --num_ref_frames 15 \
    --vid_path '/root/datasets/visdrone/transvod' \
    --resume /root/TemporalAttentionPlayground/TransVOD_plusplus/exps/msda_swinbase_full_dataset_multi_ref_frame15/checkpoint0000.pth \
    --lr_drop_epochs 4 5 \
    --num_workers 1 \
    --with_box_refine \
    --dataset_file 'vid_multi' \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.eval_e6.$T