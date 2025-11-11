#!/usr/bin/env bash
# VisDrone training WITHOUT COCO pretrained weights (train from scratch)

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/visdrone_singlebaseline_swin_384_scratch/swin_e15_bs2_numquery_300
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}

python -u main.py \
    --backbone swin_b_p4w7 \
    --epochs 15 \
    --lr_drop_epochs 10 13 \
    --num_feature_levels 1 \
    --num_queries 300 \
    --dilation \
    --batch_size 2 \
    --hidden_dim 256 \
    --num_workers 8 \
    --with_box_refine \
    --dataset_file 'visdrone_single' \
    --vid_path /root/datasets/visdrone/transvod \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
