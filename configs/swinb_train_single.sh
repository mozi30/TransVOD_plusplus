#!/usr/bin/env bash
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
  --lr 2e-4 \
  --lr_backbone 2e-5 \
  --lr_linear_proj_mult 0.1 \
  --lr_drop_epochs 10 15 \
  --num_feature_levels 1 \
  --num_queries 100 \
  --hidden_dim 256 \
  --dilation \
  --batch_size 4 \
  --num_workers 4 \
  --with_box_refine \
  --coco_pretrain \
  --resume '/home/mozi/weights/transvod/checkpoint0006.pth' \
  --vid_path '/home/mozi/datasets/visdrone/transvod' \
  --dataset_file 'vid_single' \
  --output_dir exps/msda_swinbase