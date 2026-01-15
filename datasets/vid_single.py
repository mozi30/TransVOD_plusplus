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
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms_single as T
from torch.utils.data.dataset import ConcatDataset
from datasets.vid_multi import PerturbationType, resolve_params, PerturbationSettings, PerturbSpec, Severity, PerturbationType
import random

import numpy as np
import cv2
import torch
from PIL import Image

class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1, perturbation=None):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size,)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.perturbation = perturbation
        self._rng = self.perturbation.rng()

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        # idx若为675834，则img_id为675835(img_id=idx+1)
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = self.get_image(path)

        image_id = img_id
        target = {'image_id': image_id, 'annotations': target}

        img_opencv_bgr = to_opencv_bgr(img)
        img_pertubated = self.apply_perturbation(img_opencv_bgr)
        img = to_pil_rgb(img_pertubated)

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target

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

    if image_set == 'train_vid' or image_set == "train_det" or image_set == "train_joint":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize([512], max_size=960),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([512], max_size=960),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.vid_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        # "train_joint": [(root / "Data" / "DET", root / "annotations" / 'imagenet_det_30plus1cls_vid_train.json'), (root / "Data" / "VID", root / "annotations_10true" / 'imagenet_vid_train.json')],
        #"train_det": [(root / "Data" / "DET", root / "annotations" / 'imagenet_det_30plus1cls_vid_train.json')],
        "train_vid": [(root / "Data" / "VID", root / "annotations" / 'imagenet_vid_train.json')],
        #"train_joint": [(root / "Data" / "VID", root / "annotations" / 'imagenet_vid_train.json')],  # Use train.json for joint training
        "val": [(root / "Data" / "VID", root / "annotations" / 'imagenet_vid_val.json')],
    }
    datasets = []
    for (img_folder, ann_file) in PATHS[image_set]:
        dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks, cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(), perturbation=args.perturbation)
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

    
