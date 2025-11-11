#!/usr/bin/env bash

set -x
T=$(date +%m%d%H%M)

EXP_DIR=exps/visdrone_singlebaseline_swin_384/swin_e12_bs2_numquery_100
PY_ARGS=${@:1}

python -u main.py \
    --backbone swin_b_p4w7 \
    --epochs 20 \
    --lr_drop_epochs 8 10 \
    --num_feature_levels 1 \
    --num_queries 100 \
    --dilation \
    --batch_size 4 \
    --hidden_dim 256 \
    --num_workers 8 \
    --with_box_refine \
    --resume ${EXP_DIR}/checkpoint0010.pth \
    --coco_pretrain \
    --dataset_file vid_single \
    --vid_path /root/datasets/visdrone/transvod \
    --output_dir "${EXP_DIR}" \
    --lr 5e-05 \
    --lr_backbone 5e-06 \
    --weight_decay 1e-4 \
    --clip_max_norm 0.05 \
    --seed 42 \
    --set_cost_class 2 \
    --set_cost_bbox 5 \
    --set_cost_giou 2 \
    --cls_loss_coef 1 \
    --bbox_loss_coef 5 \
    --giou_loss_coef 2 \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T