#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`
WEIGHT_DIR=/home/mozi/weights/transvod
EXP_DIR=exps/transvod/intervall-perturbations/
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

perturbations2=(
  defocus_blur
  motion_blur
  gaussian_noise
)

severities=(low med high)

interval=(
    1
    4
    16
    32
    64
)


for p in "${perturbations2[@]}"; do
    for s in "${severities[@]}"; do
        for i in "${interval[@]}"; do
        echo "Evaluating with perturbation: $p, severity: $s, interval: $i"
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
            --output_dir ${EXP_DIR} \
            --perturbation \
            --select_perturbation $p \
            --severity $s \
            --interval1 $i \
            ${PY_ARGS} 2>&1 | tee -a ${EXP_DIR}/log.eval_e6.$T
        done
    done
done

for p in "${perturbations2[@]}"; do
    for s in "${severities[@]}"; do
        for i in "${interval[@]}"; do
        echo "Evaluating with perturbation: $p, severity: $s, interval: $i"
        python -u main.py \
            --eval \
            --backbone swin_b_p4w7 \
            --num_feature_levels 1 \
            --num_queries 100 \
            --hidden_dim 256 \
            --dilation \
            --batch_size 1 \
            --num_ref_frames 3 \
            --resume ${WEIGHT_DIR}/checkpoint_ref_frame3.pth \
            --num_workers 1 \
            --with_box_refine \
            --dataset_file 'vid_multi' \
            --vid_path '/home/mozi/datasets/visdrone/transvod' \
            --output_dir ${EXP_DIR} \
            --perturbation \
            --select_perturbation $p \
            --severity $s \
            --interval1 $i \
            ${PY_ARGS} 2>&1 | tee -a ${EXP_DIR}/log.eval_e6.$T
        done
    done
done

for p in "${perturbations2[@]}"; do
    for s in "${severities[@]}"; do
        for i in "${interval[@]}"; do
        echo "Evaluating with perturbation: $p, severity: $s, interval: $i"
        python -u main.py \
            --eval \
            --backbone swin_b_p4w7 \
            --num_feature_levels 1 \
            --num_queries 100 \
            --hidden_dim 256 \
            --dilation \
            --batch_size 1 \
            --num_ref_frames 8 \
            --resume ${WEIGHT_DIR}/checkpoint_ref_frame8.pth \
            --num_workers 1 \
            --with_box_refine \
            --dataset_file 'vid_multi' \
            --vid_path '/home/mozi/datasets/visdrone/transvod' \
            --output_dir ${EXP_DIR} \
            --perturbation \
            --select_perturbation $p \
            --severity $s \
            --interval1 $i \
            ${PY_ARGS} 2>&1 | tee -a ${EXP_DIR}/log.eval_e6.$T
        done
    done
done

for p in "${perturbations2[@]}"; do
    for s in "${severities[@]}"; do
        for i in "${interval[@]}"; do
        echo "Evaluating with perturbation: $p, severity: $s, interval: $i"
        python -u main.py \
            --eval \
            --backbone swin_b_p4w7 \
            --num_feature_levels 1 \
            --num_queries 100 \
            --hidden_dim 256 \
            --dilation \
            --batch_size 1 \
            --num_ref_frames 15 \
            --resume ${WEIGHT_DIR}/checkpoint_ref_frame15.pth \
            --num_workers 1 \
            --with_box_refine \
            --dataset_file 'vid_multi' \
            --vid_path '/home/mozi/datasets/visdrone/transvod' \
            --output_dir ${EXP_DIR} \
            --perturbation \
            --select_perturbation $p \
            --severity $s \
            --interval1 $i \
            ${PY_ARGS} 2>&1 | tee -a ${EXP_DIR}/log.eval_e6.$T
        done
    done
done
