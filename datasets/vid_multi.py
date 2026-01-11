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
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
from typing import Dict, List, List, Optional

import cv2
import cv2
import numpy as np
import torch
import torch.utils.data
from pycocotools import mask as coco_mask
from traitlets import Any

from annotator import AnnotationOptions, Annotator, BBoxSpec, ImageOptions, PredFormat, gt_target_to_xyxy_pixels, target_to_xyxy_score_cls, target_to_xyxy_score_cls_single, tensor_to_pil
from .coco_video_parser import CocoVID
from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms_multi as T
from torch.utils.data.dataset import ConcatDataset
import random

import numpy as np
import cv2
import torch
from PIL import Image


class PerturbationType(str, Enum):
    GAUSSIAN_NOISE = "gaussian_noise"
    DEFOCUS_BLUR = "defocus_blur"
    MOTION_BLUR = "motion_blur"
    BRIGHTNESS_CHANGE = "brightness_change"
    CONTRAST_CHANGE = "contrast_change"
    PIXELATION = "pixelation"
    JPEG_COMPRESSION = "jpeg_compression"

class Severity(str, Enum):
    LOW = "low"
    MED = "med"
    HIGH = "high"

# ---- 3-level presets (assumes image float in [0,1]) ----
# If you use uint8 [0..255], scale Gaussian std_dev by 255, etc.
PRESETS: Dict[PerturbationType, Dict[Severity, Dict[str, Any]]] = {
    PerturbationType.GAUSSIAN_NOISE: {
        Severity.LOW:  {"std_dev": 0.01},
        Severity.MED:  {"std_dev": 0.05},
        Severity.HIGH: {"std_dev": 0.10},
    },
    PerturbationType.DEFOCUS_BLUR: {
        Severity.LOW:  {"kernel_size": 3},
        Severity.MED:  {"kernel_size": 7},
        Severity.HIGH: {"kernel_size": 11},
    },
    PerturbationType.MOTION_BLUR: {
        Severity.LOW:  {"kernel_size": 3,  "angle": 0.0},
        Severity.MED:  {"kernel_size": 7, "angle": 0.0},
        Severity.HIGH: {"kernel_size": 15, "angle": 0.0},
    },
    PerturbationType.BRIGHTNESS_CHANGE: {
        Severity.LOW:  {"factor": 1.10},
        Severity.MED:  {"factor": 1.25},
        Severity.HIGH: {"factor": 1.45},
    },
    PerturbationType.CONTRAST_CHANGE: {
        Severity.LOW:  {"factor": 1.10},
        Severity.MED:  {"factor": 1.25},
        Severity.HIGH: {"factor": 1.45},
    },
    PerturbationType.PIXELATION: {
        Severity.LOW:  {"pixel_size": 2},
        Severity.MED:  {"pixel_size": 4},
        Severity.HIGH: {"pixel_size": 6},
    },
    PerturbationType.JPEG_COMPRESSION: {
        Severity.LOW:  {"quality": 85},
        Severity.MED:  {"quality": 55},
        Severity.HIGH: {"quality": 25},
    },
}

@dataclass
class PerturbSpec:
    type: PerturbationType
    active: bool = True
    severity: Severity = Severity.MED
    p: float = 1.0           # probability to apply
    dynamic: bool = False    # if True: jitter around preset

@dataclass
class PerturbationSettings:
    enabled: bool = True
    seed: Optional[int] = None
    shuffle_order: bool = False
    specs: List[PerturbSpec] = field(default_factory=list)

    def rng(self) -> random.Random:
        return random.Random(self.seed)

    def set_active(self, ptype: PerturbationType, active: bool) -> None:
        for s in self.specs:
            if s.type == ptype:
                s.active = active

    def set_severity(self, ptype: PerturbationType, severity: Severity) -> None:
        for s in self.specs:
            if s.type == ptype:
                s.severity = severity

def resolve_params(spec: PerturbSpec, rng: random.Random) -> Dict[str, Any]:
    """Convert (type, severity) -> concrete params, with optional dynamic jitter."""
    params = dict(PRESETS[spec.type][spec.severity])

    if not spec.dynamic:
        return params

    # Small, sane jitter around presets (edit if you want wider ranges)
    if spec.type == PerturbationType.GAUSSIAN_NOISE:
        sd = params["std_dev"]
        params["std_dev"] = max(0.0, rng.uniform(0.7 * sd, 1.3 * sd))

    elif spec.type == PerturbationType.MOTION_BLUR:
        k = params["kernel_size"]
        # keep odd kernel sizes
        k2 = int(round(rng.uniform(0.8 * k, 1.2 * k)))
        params["kernel_size"] = k2 if k2 % 2 == 1 else k2 + 1
        params["angle"] = rng.uniform(0, 180)

    elif "kernel_size" in params:
        k = params["kernel_size"]
        k2 = int(round(rng.uniform(0.8 * k, 1.2 * k)))
        params["kernel_size"] = k2 if k2 % 2 == 1 else k2 + 1

    elif "strength" in params:
        s = params["strength"]
        params["strength"] = max(0.0, rng.uniform(0.7 * s, 1.3 * s))

    elif "factor" in params:
        f = params["factor"]
        params["factor"] = max(0.0, rng.uniform(0.9 * f, 1.1 * f))

    elif "pixel_size" in params:
        px = params["pixel_size"]
        params["pixel_size"] = max(1, int(round(rng.uniform(0.8 * px, 1.2 * px))))

    elif "quality" in params:
        q = params["quality"]
        params["quality"] = int(round(rng.uniform(max(1, q - 10), min(95, q + 10))))

    return params


def pick_ref_img_ids(img_ids, img_id, num_ref_frames, interval, filter_key_img=True):
    if not img_ids or num_ref_frames <= 0:
        return []

    img_ids = sorted(img_ids)
    interval = max(int(interval), 1)

    exclude = img_id if filter_key_img else None
    candidates = [x for x in img_ids if x != exclude]

    if not candidates:
        return []

    target = min(num_ref_frames, len(candidates))
    cand_set = set(candidates)

    refs = []

    # -------- interval-based sampling --------
    k = 1
    while len(refs) < target:
        left = img_id - k * interval
        right = img_id + k * interval
        added = False

        for x in (left, right):
            if len(refs) >= target:
                break
            if x in cand_set and x not in refs:
                refs.append(x)
                added = True

        if not added:
            break  # interval sampling stalled
        k += 1

    # -------- fallback: evenly spaced fill --------
    if len(refs) < target:
        remaining = [x for x in candidates if x not in refs]

        if target - len(refs) == 1:
            # evenly spaced with 1 means "best global representative"
            refs.append(remaining[len(remaining) // 2])
        else:
            need = target - len(refs)
            m = len(remaining)
            idxs = [round(i * (m - 1) / (need - 1)) for i in range(need)]

            used = set()
            for idx in idxs:
                j = idx
                step = 0
                while j in used:
                    step += 1
                    for t in (idx - step, idx + step):
                        if 0 <= t < m and t not in used:
                            j = t
                            break
                used.add(j)
                refs.append(remaining[j])

    return sorted(refs)



class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, interval1, interval2, num_ref_frames= 3,
        is_train = True,  filter_key_img=True,  cache_mode=False, local_rank=0, local_size=1, perturbation: PerturbationSettings = None):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.ann_file = ann_file
        self.frame_range = [-2, 2]
        self.num_ref_frames = num_ref_frames
        self.cocovid = CocoVID(self.ann_file)
        self.is_train = is_train
        self.filter_key_img = filter_key_img
        self.interval1 = interval1
        self.interval2 = interval2
        self.perturbation = perturbation
        self._rng = self.perturbation.rng()
        self.printed_one_time = False

        #random shuffle self.ids temporarily
        random.seed(42)
        random.shuffle(self.ids)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (images, target)
                images: Tensor [T*C, H, W] (concatenated along batch dim)
                target: dict with boxes, labels, etc. for the key frame
        """
        imgs = []

        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        video_id = img_info['video_id']

        img = self.get_image(path)
        target = {'image_id': img_id, 'annotations': target}
        img, target = self.prepare(img, target)
        imgs.append(img)

        self.random_ref_frames = False

        # ------------------------------------------------------------
        # If not a video frame (video_id == -1), just duplicate key
        # ------------------------------------------------------------
        if video_id == -1:
            for _ in range(self.num_ref_frames):
                imgs.append(img)
            print("Warning: image {} not in a video, duplicating key frame.".format(img_id))

        else:
            # All frames belonging to this video
            img_ids = self.cocovid.get_img_ids_from_vid(video_id)

            ref_img_ids = []

            # ========================================================
            # MODE B: Global random frames from the whole video
            # ========================================================
            if getattr(self, "random_ref_frames", False):
                print("Using random global ref frames...")
                # Sample uniformly from all frames in the video
                available = img_ids.copy()
                if self.filter_key_img and img_id in available:
                    available.remove(img_id)

                # If no other frames exist, fallback to duplicating key
                if len(available) == 0:
                    ref_img_ids = [img_id] * self.num_ref_frames
                else:
                    if len(available) >= self.num_ref_frames:
                        # sample without replacement
                        ref_img_ids = random.sample(available, self.num_ref_frames)
                    else:
                        # not enough unique frames → sample with replacement
                        ref_img_ids = random.choices(available, k=self.num_ref_frames)

            # ========================================================
            # MODE A: Original behavior (local / interval-based)
            # ========================================================
            else:
                if self.is_train:
                    interval = self.num_ref_frames + 2
                    left = max(img_ids[0], img_id - interval)
                    right = min(img_ids[-1], img_id + interval)
                    sample_range = list(range(left, right + 1))

                    if self.num_ref_frames >= 10:
                        sample_range = img_ids

                    if self.filter_key_img and img_id in sample_range:
                        sample_range.remove(img_id)

                    # Ensure we have enough to sample from
                    while len(sample_range) < self.num_ref_frames:
                        sample_range.extend(sample_range)

                    ref_img_ids = random.sample(sample_range, self.num_ref_frames)

                else:
                    ref_img_ids = pick_ref_img_ids(img_ids, img_id, self.num_ref_frames, self.interval1, self.filter_key_img)
                    if( not self.printed_one_time):
                        print("Using pick_ref_img_ids with interval {} to select reference frames.".format(self.interval1))
                        print("Image ID: {}, Video ID: {}, Available frames: {}, Selected ref frames: {}".format(img_id, video_id, img_ids, ref_img_ids))   
                        self.printed_one_time = True
                    # ref_img_ids = []
                    # Len = len(img_ids)
                    # interval = max(int(Len // self.interval1), 1)

                    # if self.num_ref_frames < 8:
                    #     left_indexs = int((img_id - img_ids[0]) // interval)
                    #     right_indexs = int((img_ids[-1] - img_id) // interval)
                    #     if left_indexs < self.num_ref_frames:
                    #         for i in range(self.num_ref_frames):
                    #             ref_img_ids.append(min(img_id + (i + 1) * interval, img_ids[-1]))
                    #     else:
                    #         for i in range(self.num_ref_frames):
                    #             ref_img_ids.append(max(img_id - (i + 1) * interval, img_ids[0]))

                    # sample_range = []
                    # if self.num_ref_frames >= 8:
                    #     left_indexs = int((img_ids[0] - img_id) // interval)
                    #     right_indexs = int((img_ids[-1] - img_id) // interval)
                    #     for i in range(left_indexs, right_indexs):
                    #         if i < 0:
                    #             index = max(img_id + i * interval, img_ids[0])
                    #             sample_range.append(index)
                    #         elif i > 0:
                    #             index = min(img_id + i * interval, img_ids[-1])
                    #             sample_range.append(index)
                    #     if self.filter_key_img and img_id in sample_range:
                    #         sample_range.remove(img_id)

                    #     # Ensure unique samples
                    #     unique_samples = set(sample_range)
                    #     while len(unique_samples) < self.num_ref_frames:
                    #         interval += 1  # Increase interval to find more unique samples
                    #         sample_range = []
                    #         left_indexs = int((img_ids[0] - img_id) // interval)
                    #         right_indexs = int((img_ids[-1] - img_id) // interval)
                    #         for i in range(left_indexs, right_indexs):
                    #             if i < 0:
                    #                 index = max(img_id + i * interval, img_ids[0])
                    #                 sample_range.append(index)
                    #             elif i > 0:
                    #                 index = min(img_id + i * interval, img_ids[-1])
                    #                 sample_range.append(index)
                    #         unique_samples = set(sample_range)

                    #     # Ensure we have the required number of unique samples
                    #     while len(unique_samples) < self.num_ref_frames:
                    #         closest = min(unique_samples, key=lambda x: abs(x - img_id))
                    #         unique_samples.add(closest)

                    #     ref_img_ids = sorted(unique_samples)[:self.num_ref_frames]
                    #     ref_img_ids.sort()

            # --------------------------------------------------------
            # Load reference images
            # --------------------------------------------------------
            for ref_img_id in ref_img_ids:
                ref_ann_ids = coco.getAnnIds(imgIds=ref_img_id)
                ref_img_info = coco.loadImgs(ref_img_id)[0]
                ref_img_path = ref_img_info['file_name']
                ref_img = self.get_image(ref_img_path)
                imgs.append(ref_img)



        for i, img in enumerate(imgs):
            img_opencv_bgr = to_opencv_bgr(img)
            img_pertubated = self.apply_perturbation(img_opencv_bgr)
            img = to_pil_rgb(img_pertubated)
            imgs[i] = img

        # annotator = Annotator(
        #     image_opts=ImageOptions(channel_order="rgb"),
        #     ann_opts=AnnotationOptions(
        #         format=PredFormat(
        #             bbox_start=0,
        #             bbox_len=4,
        #             bbox_spec=BBoxSpec(fmt="xyxy", normalized=False, exclusive_max=True),
        #             score_index=4,
        #             cls_index=5,
        #         ),
        #         class_names=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        #     ),
        #     preset="filled",
        #     output_dir=os.path.join("exps/transvod/pertubation/", "annotated_getitem")
        # )

        # annotator2 = Annotator(
        #     image_opts=ImageOptions(channel_order="rgb"),
        #     ann_opts=AnnotationOptions(
        #         format=PredFormat(
        #             bbox_start=0,
        #             bbox_len=4,
        #             bbox_spec=BBoxSpec(fmt="xyxy", normalized=False, exclusive_max=True),
        #             score_index=4,
        #             cls_index=5,
        #         ),
        #         class_names=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        #     ),
        #     preset="filled",
        #     output_dir=os.path.join("exps/transvod/pertubation/", "annotated_transform")
        # )

        # target_draw = target_to_xyxy_score_cls_single(target)
        # result_draw = []  # No predictions, only ground truth
        # img = annotator.draw(imgs[0], preds=result_draw, targets=target_draw, score_threshold=0.1)
        # annotator.save(image=img, name=f"{target['image_id'].item():012d}.jpg")


        if self._transforms is not None:
            imgs, target = self._transforms(imgs, target) 

        # h, w = target["size"].tolist() 
        # img_t = imgs[0]                  # CHW
        # img_t = img_t[:, :h, :w]       
        # imgs_pil = tensor_to_pil(imgs)
        

        # target_draw = gt_target_to_xyxy_pixels(target)
        # result_draw = []  # No predictions, only ground truth
        # img = annotator2.draw(imgs_pil, preds=result_draw, targets=target_draw, score_threshold=0.1)
        # annotator2.save(image=img, name=f"{target['image_id'].item():012d}.jpg")
        return  torch.cat(imgs, dim=0),  target
    
    
    
    def apply_perturbation(self, img):
        if self.perturbation.enabled:

            specs = [s for s in self.perturbation.specs if s.active]
            if self.perturbation.shuffle_order:
                self._rng.shuffle(specs)

            for s in specs:
                if self._rng.random() > s.p:
                    continue
                params = resolve_params(s, self._rng)
                typ = s.type
                if typ == PerturbationType.GAUSSIAN_NOISE:
                    std_dev = params.get("std_dev", 0.05)
                    noise = np.random.normal(0, std_dev * 255, img.shape).astype(np.float32)
                    img = img.astype(np.float32) + noise
                    img = np.clip(img, 0, 255).astype(np.uint8)
                    return img
                if typ == PerturbationType.DEFOCUS_BLUR:
                    kernel_size = params.get("kernel_size", 7)
                    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
                    cv2.circle(kernel, (kernel_size // 2, kernel_size // 2), kernel_size // 2, 1, -1)
                    kernel /= np.sum(kernel)
                    img = cv2.filter2D(img, -1, kernel)
                    return img
                if typ == PerturbationType.MOTION_BLUR:
                    kernel_size = params.get("kernel_size", 15)
                    angle = params.get("angle", 0.0)
                    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
                    xs = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
                    ys = np.tan(np.deg2rad(angle)) * xs
                    for i in range(kernel_size):
                        x = int(xs[i] + kernel_size // 2)
                        y = int(ys[i] + kernel_size // 2)
                        if 0 <= x < kernel_size and 0 <= y < kernel_size:
                            kernel[y, x] = 1
                    kernel /= np.sum(kernel)
                    img = cv2.filter2D(img, -1, kernel)
                    return img
                if typ == PerturbationType.BRIGHTNESS_CHANGE:
                    factor = params.get("factor", 1.25)
                    img = img.astype(np.float32) * factor
                    img = np.clip(img, 0, 255).astype(np.uint8)
                    return img
                if typ == PerturbationType.CONTRAST_CHANGE:
                    factor = params.get("factor", 1.25)
                    mean = np.mean(img, axis=(0, 1), keepdims=True)
                    img = (img.astype(np.float32) - mean) * factor + mean
                    img = np.clip(img, 0, 255).astype(np.uint8)
                    return img
                if typ == PerturbationType.PIXELATION:
                    pixel_size = params.get("pixel_size", 8)
                    h, w = img.shape[:2]
                    temp = cv2.resize(img, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
                    img = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
                    return img
                if typ == PerturbationType.JPEG_COMPRESSION:
                    quality = params.get("quality", 50)
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                    _, encimg = cv2.imencode('.jpg', img, encode_param)
                    img = cv2.imdecode(encimg, 1)
                    return img
                raise ValueError(f"Unsupported perturbation type: {typ}")
                  # If your pipeline expects CHW afterward, transpose back:
        return img


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        classes -= 1  # change to 0-based index

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        
        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train_vid' or image_set == "train_det" or image_set == "train_joint":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize([544], max_size=1000),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([544], max_size=1000),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.vid_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        #"train_det": [(root / "Data" / "DET", root / "annotations" / 'imagenet_det_30plus1cls_vid_train.json')],
        "train_vid": [(root / "Data" / "VID", root / "annotations" / 'imagenet_vid_train.json')],
        #"train_joint": [(root / "Data" , root / "annotations" / 'imagenet_vid_train_joint_30.json')],
        "val": [(root / "Data" / "VID", root / "annotations" / 'imagenet_vid_val.json')],
    }
    datasets = []
    for (img_folder, ann_file) in PATHS[image_set]:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), is_train =(not args.eval), interval1=args.interval1,
                                interval2=args.interval2, num_ref_frames = args.num_ref_frames, return_masks=args.masks, cache_mode=args.cache_mode, 
                                local_rank=get_local_rank(), local_size=get_local_size(), perturbation=args.perturbation)
        datasets.append(dataset)
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)

    
def to_opencv_bgr(img):
    """
    Accepts:
      - PIL.Image (RGB)
      - np.ndarray (HWC or CHW, RGB or BGR)
      - torch.Tensor (CHW or HWC, RGB, float or uint8)

    Returns:
      - np.ndarray, uint8, HWC, BGR, [0..255]
    """

    # --- PIL Image (RGB, HWC) ---
    if Image is not None and isinstance(img, Image.Image):
        img = np.array(img)                 # HWC, RGB, uint8
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    # --- Torch Tensor ---
    if torch is not None and isinstance(img, torch.Tensor):
        img = img.detach().cpu()

        # CHW -> HWC
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = img.permute(1, 2, 0)

        img = img.numpy()

        # float -> uint8
        if img.dtype != np.uint8:
            img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)

        # RGB -> BGR
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    # --- NumPy array ---
    if isinstance(img, np.ndarray):

        # CHW -> HWC
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))

        # float -> uint8
        if img.dtype != np.uint8:
            img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)

        # Assume RGB -> BGR if 3 channels
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    raise TypeError(f"Unsupported image type: {type(img)}")

def to_pil_rgb(img):
    """
    Accepts:
      - PIL.Image
      - np.ndarray (HWC or CHW, RGB or BGR)
      - torch.Tensor (CHW or HWC, RGB, float or uint8)

    Returns:
      - PIL.Image.Image (RGB)
    """

    # --- Already PIL ---
    if Image is not None and isinstance(img, Image.Image):
        return img.convert("RGB")

    # --- Torch Tensor ---
    if torch is not None and isinstance(img, torch.Tensor):
        img = img.detach().cpu()

        # CHW -> HWC
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = img.permute(1, 2, 0)

        img = img.numpy()

        # float -> uint8
        if img.dtype != np.uint8:
            img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)

        # Assume BGR -> RGB if 3 channels
        if img.ndim == 3 and img.shape[2] == 3:
            img = img[..., ::-1]

        return Image.fromarray(img, mode="RGB")

    # --- NumPy array ---
    if isinstance(img, np.ndarray):

        # CHW -> HWC
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))

        # float -> uint8
        if img.dtype != np.uint8:
            img = np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)

        # Assume BGR -> RGB if 3 channels
        if img.ndim == 3 and img.shape[2] == 3:
            img = img[..., ::-1]

        return Image.fromarray(img, mode="RGB")

    raise TypeError(f"Unsupported image type: {type(img)}")
