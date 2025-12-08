#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`
source /root/activate_msda.sh
EXP_DIR=exps/msda_swinbase
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --eval \
    --backbone swin_b_p4w7 \
    --epochs 7 \
    --lr_drop_epochs 5 6 \
    --num_feature_levels 1\
    --num_queries 100 \
    --dilation \
    --batch_size 64 \
    --hidden_dim 256 \
    --num_workers 1 \
    --with_box_refine \
    --vid_path '/root/datasets/visdrone/transvod' \
    --resume /root/TemporalAttentionPlayground/TransVOD_plusplus/exps/msda_swinbase_full_dataset/checkpoint0020.pth \
    --dataset_file 'vid_single' \
    --output_dir 'exps/msda_swinbase'
