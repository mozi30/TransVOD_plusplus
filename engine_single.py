# Modified by Qianyu Zhou and Lu He
# ------------------------------------------------------------------------
# TransVOD++
# Copyright (c) 2022 Shanghai Jiao Tong University. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) SenseTime. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
from annotator import AnnotationOptions, Annotator, BBoxSpec, ImageOptions, PredFormat
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher_single import data_prefetcher

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    print("------------------------------------------------------!!!!")
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    # print("samples", prefecher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        # samples, ref_samples, targets = prefetcher.next()
        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
# def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
#     model.eval()
#     criterion.eval()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
#     header = 'Test:'

#     iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
#     coco_evaluator = CocoEvaluator(base_ds, iou_types)
#     # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

#     panoptic_evaluator = None
#     if 'panoptic' in postprocessors.keys():
#         panoptic_evaluator = PanopticEvaluator(
#             data_loader.dataset.ann_file,
#             data_loader.dataset.ann_folder,
#             output_dir=os.path.join(output_dir, "panoptic_eval"),
#         )

#     for samples, targets in metric_logger.log_every(data_loader, 10, header):
#         samples = samples.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         outputs = model(samples)
#         loss_dict = criterion(outputs, targets)
#         weight_dict = criterion.weight_dict
#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                     for k, v in loss_dict_reduced.items() if k in weight_dict}
#         loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#                                       for k, v in loss_dict_reduced.items()}
#         metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
#                              **loss_dict_reduced_scaled,
#                              **loss_dict_reduced_unscaled)
#         metric_logger.update(class_error=loss_dict_reduced['class_error'])

#         orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
#         results = postprocessors['bbox'](outputs, orig_target_sizes)
#         if 'segm' in postprocessors.keys():
#             target_sizes = torch.stack([t["size"] for t in targets], dim=0)
#             results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
#         res = {target['image_id'].item(): output for target, output in zip(targets, results)}
#         if coco_evaluator is not None:
#             coco_evaluator.update(res)

#         if panoptic_evaluator is not None:
#             res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
#             for i, target in enumerate(targets):
#                 image_id = target["image_id"].item()
#                 file_name = f"{image_id:012d}.png"
#                 res_pano[i]["image_id"] = image_id
#                 res_pano[i]["file_name"] = file_name

#             panoptic_evaluator.update(res_pano)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     if coco_evaluator is not None:
#         coco_evaluator.synchronize_between_processes()
#     if panoptic_evaluator is not None:
#         panoptic_evaluator.synchronize_between_processes()

#     # accumulate predictions from all images
#     if coco_evaluator is not None:
#         coco_evaluator.accumulate()
#         coco_evaluator.summarize()
#     panoptic_res = None
#     if panoptic_evaluator is not None:
#         panoptic_res = panoptic_evaluator.summarize()
#     stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#     if coco_evaluator is not None:
#         if 'bbox' in postprocessors.keys():
#             stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
#         if 'segm' in postprocessors.keys():
#             stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
#     if panoptic_res is not None:
#         stats['PQ_all'] = panoptic_res["All"]
#         stats['PQ_th'] = panoptic_res["Things"]
#         stats['PQ_st'] = panoptic_res["Stuff"]
#     return stats, coco_evaluator

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, inference=False, sample_size=-1):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    annotator = Annotator(
        image_opts=ImageOptions(channel_order="rgb"),
        ann_opts=AnnotationOptions(
            format=PredFormat(
                bbox_start=0,
                bbox_len=4,
                bbox_spec=BBoxSpec(fmt="xyxy", normalized=False),
                score_index=4,
                cls_index=5,
            ),
            class_names=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        ),
        preset="filled",
    )

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # # ---------- DEBUG BLOCK: raw preds vs targets (normalized) ----------
        # logits = outputs['pred_logits'][0]   # [num_queries, num_classes+1]
        # boxes  = outputs['pred_boxes'][0]    # [num_queries, 4] cxcywh in [0,1]

        # prob = logits.softmax(-1)
        # scores, labels = prob[..., :-1].max(-1)

        # topk_scores, topk_idx = scores.topk(10)
        # topk_labels = labels[topk_idx]
        # topk_boxes  = boxes[topk_idx]

        # print("\n================ RAW PREDICTIONS (normalized) ================")
        # for i in range(topk_scores.numel()):
        #     s = topk_scores[i].item()
        #     c = topk_labels[i].item()
        #     b = topk_boxes[i].tolist()
        #     print(f"Pred {i:02d}: cls={c}, score={s:.3f}, "
        #           f"cxcywh_norm=[{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}]")

        # tgt = targets[0]
        # gt_labels = tgt["labels"]
        # gt_boxes  = tgt["boxes"]

        # print("\n================ GT (normalized) ================")
        # print(f"num_gt_boxes: {gt_labels.numel()}")
        # for i in range(gt_labels.numel()):
        #     c = gt_labels[i].item()
        #     b = gt_boxes[i].tolist()
        #     print(f"GT  {i:02d}: cls={c}, cxcywh_norm="
        #           f"[{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}]")

        # from util import box_ops
        # ious_norm, _ = box_ops.box_iou(
        #     box_ops.box_cxcywh_to_xyxy(topk_boxes),
        #     box_ops.box_cxcywh_to_xyxy(gt_boxes),
        # )
        # print("\nIoU (normalized space, preds x gt):")
        # print(ious_norm)

        # # ---------- NEW PART: check postprocessor / absolute coords ----------
        # orig_h, orig_w = tgt["orig_size"].tolist()  # [H, W]
        # print("\norig_size (H, W):", tgt["orig_size"].tolist())

        # # apply the same bbox postprocessor used for eval
        # results = postprocessors['bbox'](outputs, torch.stack([t["orig_size"] for t in targets], dim=0))
        # res0 = results[0]  # first image

        # pred_boxes_abs = res0["boxes"]      # [N_det, 4] xyxy in absolute pixels
        # pred_scores_abs = res0["scores"]
        # pred_labels_abs = res0["labels"]    # these are category indices used for COCO

        # # take top 10 by score AFTER postprocessing
        # topk_scores2, topk_idx2 = pred_scores_abs.topk(min(10, pred_scores_abs.numel()))
        # topk_boxes_abs = pred_boxes_abs[topk_idx2]
        # topk_labels_abs = pred_labels_abs[topk_idx2]

        # print("\n================ POSTPROCESSED PREDICTIONS (abs xyxy) ================")
        # for i in range(topk_scores2.numel()):
        #     s = topk_scores2[i].item()
        #     c = topk_labels_abs[i].item()
        #     x1, y1, x2, y2 = topk_boxes_abs[i].tolist()
        #     print(f"Pred_pp {i:02d}: cat={c}, score={s:.3f}, xyxy_abs="
        #           f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

        # # build GT in abs xyxy using orig_size (assuming gt_boxes are normalized cxcywh)
        # gt_xyxy_abs = box_ops.box_cxcywh_to_xyxy(gt_boxes)
        # # scale normalized [0,1] -> pixels
        # scale = gt_xyxy_abs.new_tensor([orig_w, orig_h, orig_w, orig_h])
        # gt_xyxy_abs = gt_xyxy_abs * scale

        # print("\n================ GT (abs xyxy) ================")
        # for i in range(gt_labels.numel()):
        #     x1, y1, x2, y2 = gt_xyxy_abs[i].tolist()
        #     print(f"GT_abs {i:02d}: cls={gt_labels[i].item()}, xyxy_abs="
        #           f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

        # # IoU in absolute pixel space (what COCO evaluator uses)
        # ious_abs, _ = box_ops.box_iou(topk_boxes_abs, gt_xyxy_abs)
        # print("\nIoU (absolute pixel space, preds_pp x gt):")
        # print(ious_abs)

        # # Stop after first batch while debugging
        # exit(0)
        # # ---------- END DEBUG BLOCK ----------

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
