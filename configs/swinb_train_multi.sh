#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`
source /home/mozi/activate msda
EXP_DIR=/root/TemporalAttentionPlayground/TransVOD_plusplus/exps/msda_swinbase_full_dataset_multi
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone swin_b_p4w7 \
    --epochs 10 \
    --num_feature_levels 1 \
    --lr_drop_epochs 1 3 \
    --num_queries 100 \
    --hidden_dim 256 \
    --dilation \
    --batch_size 1 \
    --num_workers 1 \
    --num_ref_frames 8 \
    --resume /root/TemporalAttentionPlayground/TransVOD_plusplus/exps/msda_swinbase_full_dataset/checkpoint0008.pth \
    --with_box_refine \
    --vid_path '/root/datasets/visdrone/transvod' \
    --dataset_file 'vid_multi' \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T