#!/usr/bin/env bash
export CUDA_LAUNCH_BLOCKING=1
set -x
T=`date +%m%d%H%M`
source /home/mozi/activate_msda.sh
EXP_DIR=exps/msda_swinbase
PY_ARGS=${@:1}

# ensure log dir exists so tee won't fail
mkdir -p "${EXP_DIR}"

python3 -u main.py \
    --backbone swin_b_p4w7 \
    --epochs 20 \
    --num_feature_levels 1 \
    --lr_drop_epochs 5 6 \
    --num_queries 100 \
    --hidden_dim 256 \
    --dilation \
    --batch_size 1 \
    --num_workers 1 \
    --with_box_refine \
    --coco_pretrain \
    --pretrained '/home/mozi/weights/transvod/checkpoint0006.pth' \
    --vid_path '/home/mozi/datasets/visdrone/transvod' \
    --dataset_file 'vid_single' \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T