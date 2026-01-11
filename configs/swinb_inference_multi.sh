#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`
WEIGHT_DIR=/home/mozi/weights/transvod
EXP_DIR=exps/transvod/pertubation/
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --eval \
    --backbone swin_b_p4w7 \
    --num_feature_levels 1 \
    --num_queries 100 \
    --hidden_dim 256 \
    --dilation \
    --batch_size 1 \
    --num_ref_frames 1 \
    --resume ${WEIGHT_DIR}/checkpoint_ref_frame1.pth \
    --num_workers 1 \
    --with_box_refine \
    --dataset_file 'vid_multi' \
    --vid_path '/home/mozi/datasets/visdrone/transvod' \
    --perturbation \
    --select_perturbation 'motion_blur' \
    --severity 'high' \
    --output_dir ${EXP_DIR} \
    --inference \
    --sample_size 50 \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.eval_e6.$T
