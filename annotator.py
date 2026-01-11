from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from typing import Any


# ============================================================
# 1) IMAGE SETTINGS (input type handling + per-type options)
# ============================================================

@dataclass(frozen=True)
class ImageOptions:
    """
    Controls how images are interpreted and returned.

    - channel_order:
        For numpy/torch inputs, you can tell the annotator whether channels are RGB or BGR.
        (PIL is always treated as RGB/RGBA internally.)

    - assume_chw_for_tensors / assume_chw_for_numpy:
        If True, treat (C,H,W) as the default layout when ambiguous.

    - float_range:
        For float images, how to interpret values.
        "auto": if max<=1 -> [0,1] else [0,255]
        "0_1": always [0,1]
        "0_255": always [0,255]

    - return_same_type:
        Return the same container type as input (PIL->PIL, numpy->numpy, torch->torch).
        (This is usually what you want.)

    - preserve_pil_mode:
        Try to preserve original PIL mode when converting back.
    """
    channel_order: str = "rgb"  # "rgb" or "bgr" (only matters for numpy/torch)
    assume_chw_for_tensors: bool = True
    assume_chw_for_numpy: bool = False
    float_range: str = "auto"  # "auto" | "0_1" | "0_255"
    return_same_type: bool = True
    preserve_pil_mode: bool = True


# ============================================================
# 2) ANNOTATION SETTINGS (base formats + explicit index map)
# ============================================================

@dataclass(frozen=True)
class BBoxSpec:
        """Describe how bbox is stored in a prediction.

        fmt:
            - "xyxy": [x1,y1,x2,y2]
            - "xywh": [x,y,w,h] (top-left)
            - "cxcywh": [cx,cy,w,h] (center)

        normalized:
            - True if bbox coords are in [0,1] relative to image W/H
            - False if bbox coords are in pixels

        exclusive_max:
            - If True, interpret the max corner (x2,y2) as an exclusive boundary
                (common when converting from COCO xywh via x2=x+w, y2=y+h).
                For drawing in pixel indices, we convert to inclusive by subtracting 1.
        """

        fmt: str = "xyxy"
        normalized: bool = False
        exclusive_max: bool = False


@dataclass(frozen=True)
class PredFormat:
    """
    Describe how to read ONE prediction item.

    This supports common “flat array” representations by telling us:
      - where bbox starts (bbox_start) and bbox length (bbox_len)
      - where score/confidence is (score_index) (optional)
      - where class id is (cls_index) (optional)
      - where label string is (label_index) (optional)

    Example: YOLO-ish: [x1,y1,x2,y2, conf, cls]
      bbox_start=0, bbox_len=4, score_index=4, cls_index=5

    You can also provide a dict-based format via DictPredFormat below.
    """
    bbox_start: int = 0
    bbox_len: int = 4
    bbox_spec: BBoxSpec = field(default_factory=BBoxSpec)

    score_index: Optional[int] = None
    cls_index: Optional[int] = None
    label_index: Optional[int] = None

    # Optional: if your array has a "class probability vector" starting somewhere
    # and you want score=prob[max_cls], cls=argmax(probs)
    probs_start: Optional[int] = None
    probs_len: Optional[int] = None


@dataclass(frozen=True)
class DictPredFormat:
    """
    Dict prediction mapping: specify which keys to look for.

    - bbox_keys: keys checked in order for bbox array
    - score_keys: keys checked in order for score/confidence
    - cls_keys: keys checked in order for class id
    - label_keys: keys checked in order for label/name
    """
    bbox_keys: Tuple[str, ...] = ("box", "bbox", "xyxy")
    score_keys: Tuple[str, ...] = ("score", "conf", "confidence")
    cls_keys: Tuple[str, ...] = ("cls", "class", "category_id")
    label_keys: Tuple[str, ...] = ("label", "name")
    bbox_spec: BBoxSpec = field(default_factory=BBoxSpec)


@dataclass(frozen=True)
class AnnotationOptions:
    """
    Controls how predictions are interpreted and how labels are resolved.

    - format:
        Either a PredFormat (array-like) or DictPredFormat (dict-like).
    - class_names:
        Optional mapping from cls -> name if label is missing.
    """
    format: Union[PredFormat, DictPredFormat] = field(default_factory=PredFormat)
    class_names: Optional[Sequence[str]] = None


# ============================================================
# 3) DRAWING PRESETS (how annotation is built visually)
# ============================================================

PRESETS: Dict[str, Dict[str, Any]] = {
    "classic": dict(
        box=True,
        box_style="rect",       # "rect" or "corners"
        thickness=3,

        box_fill=False,
        box_fill_alpha=60,

        label=True,
        label_position="tl",    # tl,tr,bl,br
        label_bg=True,
        label_bg_alpha=220,
        text_padding=4,
        font_size=16,

        show_label=True,
        show_score=True,
        show_class=False,
        score_fmt="{:.2f}",
        sep=" ",

        color_scheme="hash",    # "hash" | "mono" | "lime"
        color=None,             # override RGB tuple
        text_shadow=False,
        text_shadow_px=2,
    ),
    "filled": dict(
        box=True,
        box_style="rect",
        thickness=3,

        box_fill=True,
        box_fill_alpha=60,

        label=True,
        label_position="tl",
        label_bg=True,
        label_bg_alpha=220,
        text_padding=4,
        font_size=16,

        show_label=True,
        show_score=True,
        show_class=False,
        score_fmt="{:.2f}",
        sep=" ",

        color_scheme="hash",
        color=None,
        text_shadow=False,
        text_shadow_px=2,
    ),
    "minimal": dict(box=True, label=False, thickness=2, color_scheme="hash", color=None),
    "corners": dict(box=True, box_style="corners", corner_frac=0.22, label=True, thickness=3, color_scheme="hash", color=None),
    "neon": dict(box=True, label=True, thickness=5, color_scheme="lime", label_bg=False, text_shadow=True, text_shadow_px=2),
}


# ============================================================
# 4) IMPLEMENTATION
# ============================================================

def _is_torch_tensor(x: Any) -> bool:
    return x.__class__.__module__.startswith("torch") and hasattr(x, "detach") and hasattr(x, "cpu")


def _to_numpy(image: Any) -> Tuple[np.ndarray, str, dict]:
    if isinstance(image, Image.Image):
        return np.array(image), "pil", {"mode": image.mode}

    if isinstance(image, np.ndarray):
        return image, "numpy", {"dtype": image.dtype}

    if _is_torch_tensor(image):
        t = image.detach().cpu()
        return t.numpy(), "torch", {"device": getattr(image, "device", None), "dtype": getattr(image, "dtype", None)}

    raise TypeError("Unsupported image type. Use PIL.Image, numpy.ndarray, or torch.Tensor.")


def _from_numpy(arr: np.ndarray, kind: str, meta: dict, image_opts: ImageOptions) -> Any:
    if not image_opts.return_same_type:
        return arr  # always numpy out

    if kind == "pil":
        if image_opts.preserve_pil_mode:
            mode = meta.get("mode")
            if mode:
                try:
                    return Image.fromarray(arr, mode=mode)
                except Exception:
                    pass
        return Image.fromarray(arr)

    if kind == "torch":
        import torch  # type: ignore
        out = torch.from_numpy(arr)
        device = meta.get("device")
        if device is not None:
            try:
                out = out.to(device)
            except Exception:
                pass
        return out

    return arr


def _ensure_uint8_hwc(arr: np.ndarray, image_opts: ImageOptions, kind: str) -> Tuple[np.ndarray, dict]:
    """
    Normalize to uint8 HWC RGB(A) for PIL drawing.
    Returns (arr_u8_hwc_rgb, meta_for_restore).
    """
    meta: dict = {
        "original_dtype": arr.dtype,
        "original_ndim": arr.ndim,
        "was_chw": False,
        "was_grayscale": False,
        "channel_order": image_opts.channel_order.lower(),
        "float_assumed_01": False,
    }

    # grayscale
    if arr.ndim == 2:
        meta["was_grayscale"] = True
        arr = arr[:, :, None]

    # Decide if CHW
    assume_chw = image_opts.assume_chw_for_tensors if kind == "torch" else image_opts.assume_chw_for_numpy
    if arr.ndim == 3:
        if assume_chw and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
            meta["was_chw"] = True
            arr = np.transpose(arr, (1, 2, 0))
        else:
            # Heuristic: if looks like CHW, convert
            if arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
                meta["was_chw"] = True
                arr = np.transpose(arr, (1, 2, 0))

    # dtype -> uint8
    if np.issubdtype(arr.dtype, np.floating):
        fr = image_opts.float_range
        arr_f = np.array(arr, dtype=np.float32, copy=False)

        if fr == "0_1":
            meta["float_assumed_01"] = True
            arr_f = np.clip(arr_f, 0.0, 1.0) * 255.0
        elif fr == "0_255":
            arr_f = np.clip(arr_f, 0.0, 255.0)
        else:  # auto
            mx = float(np.nanmax(arr_f)) if arr_f.size else 0.0
            if mx <= 1.0:
                meta["float_assumed_01"] = True
                arr_f = np.clip(arr_f, 0.0, 1.0) * 255.0
            else:
                arr_f = np.clip(arr_f, 0.0, 255.0)

        arr_u8 = np.rint(arr_f).astype(np.uint8)
    elif arr.dtype == np.uint8:
        arr_u8 = arr
    else:
        arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)

    # BGR->RGB for drawing
    if meta["channel_order"] == "bgr" and arr_u8.ndim == 3 and arr_u8.shape[2] in (3, 4):
        if arr_u8.shape[2] == 3:
            arr_u8 = arr_u8[:, :, ::-1]
        else:
            b, g, r, a = np.split(arr_u8, 4, axis=2)
            arr_u8 = np.concatenate([r, g, b, a], axis=2)

    return arr_u8, meta


def _restore_from_uint8_hwc(arr_u8_rgb: np.ndarray, meta: dict) -> np.ndarray:
    out = arr_u8_rgb

    # RGB->BGR if needed
    if meta.get("channel_order") == "bgr" and out.ndim == 3 and out.shape[2] in (3, 4):
        if out.shape[2] == 3:
            out = out[:, :, ::-1]
        else:
            r, g, b, a = np.split(out, 4, axis=2)
            out = np.concatenate([b, g, r, a], axis=2)

    # HWC->CHW if needed
    if meta.get("was_chw"):
        out = np.transpose(out, (2, 0, 1))

    # grayscale
    if meta.get("was_grayscale"):
        out = out[:, :, 0]

    # restore dtype (best-effort)
    dtype = meta.get("original_dtype", np.uint8)
    if dtype == np.uint8:
        return out

    if np.issubdtype(dtype, np.floating):
        if meta.get("float_assumed_01"):
            return (out.astype(np.float32) / 255.0).astype(dtype, copy=False)
        return out.astype(dtype, copy=False)

    return out.astype(dtype, copy=False)


def _color_for_class(cls: Optional[int], scheme: str = "hash") -> Tuple[int, int, int]:
    if scheme == "mono":
        return (255, 255, 255)
    if scheme == "lime":
        return (0, 255, 0)
    if cls is None:
        return (0, 255, 0)

    x = (int(cls) * 2654435761) & 0xFFFFFFFF
    r = int(80 + ((x & 0xFF) / 255.0) * 175)
    g = int(80 + (((x >> 8) & 0xFF) / 255.0) * 175)
    b = int(80 + (((x >> 16) & 0xFF) / 255.0) * 175)
    return (r, g, b)


def _load_font(path: Optional[str], size: int) -> ImageFont.ImageFont:
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _bbox_to_xyxy(b: Sequence[float], spec: BBoxSpec, w: int, h: int) -> Tuple[float, float, float, float]:
    if len(b) != 4:
        raise ValueError(f"bbox must have 4 numbers, got {len(b)}")

    x, y, z, t = map(float, b)
    if spec.normalized:
        # scale normalized to pixels
        if spec.fmt == "xyxy":
            x1, y1, x2, y2 = x * w, y * h, z * w, t * h
        elif spec.fmt == "xywh":
            x1, y1, bw, bh = x * w, y * h, z * w, t * h
            x2, y2 = x1 + bw, y1 + bh
        elif spec.fmt == "cxcywh":
            cx, cy, bw, bh = x * w, y * h, z * w, t * h
            x1, y1 = cx - bw / 2.0, cy - bh / 2.0
            x2, y2 = cx + bw / 2.0, cy + bh / 2.0
        else:
            raise ValueError(f"Unknown bbox fmt: {spec.fmt}")
    else:
        if spec.fmt == "xyxy":
            x1, y1, x2, y2 = x, y, z, t
        elif spec.fmt == "xywh":
            x1, y1, bw, bh = x, y, z, t
            x2, y2 = x1 + bw, y1 + bh
        elif spec.fmt == "cxcywh":
            cx, cy, bw, bh = x, y, z, t
            x1, y1 = cx - bw / 2.0, cy - bh / 2.0
            x2, y2 = cx + bw / 2.0, cy + bh / 2.0
        else:
            raise ValueError(f"Unknown bbox fmt: {spec.fmt}")

    # If max corner is stored as exclusive (x2/y2 are one past the last pixel),
    # convert to inclusive for raster drawing.
    if spec.exclusive_max:
        x2 -= 1.0
        y2 -= 1.0

    # sort + clamp later
    return x1, y1, x2, y2


def _parse_pred(
    p: Any,
    fmt: Union[PredFormat, DictPredFormat],
    img_w: int,
    img_h: int,
    class_names: Optional[Sequence[str]],
) -> Tuple[Tuple[float, float, float, float], Optional[float], Optional[int], Optional[str]]:
    """
    Returns: (xyxy pixels), score, cls, label
    """
    if isinstance(fmt, DictPredFormat):
        if not isinstance(p, Mapping):
            raise TypeError("DictPredFormat requires dict-like predictions.")

        bbox_val = None
        for k in fmt.bbox_keys:
            if k in p:
                bbox_val = p[k]
                break
        if bbox_val is None:
            raise ValueError(f"Dict prediction missing bbox keys {fmt.bbox_keys}")

        score = None
        for k in fmt.score_keys:
            if k in p and p[k] is not None:
                score = float(p[k])
                break

        cls = None
        for k in fmt.cls_keys:
            if k in p and p[k] is not None:
                cls = int(p[k])
                break

        label = None
        for k in fmt.label_keys:
            if k in p and p[k] is not None:
                label = str(p[k])
                break

        x1, y1, x2, y2 = _bbox_to_xyxy(bbox_val, fmt.bbox_spec, img_w, img_h)
        if (not label) and class_names is not None and cls is not None and 0 <= cls < len(class_names):
            label = class_names[cls]
        return (x1, y1, x2, y2), score, cls, label

    # PredFormat (array-like)
    if isinstance(p, (list, tuple, np.ndarray)):
        v = list(p)

        # class probs vector option
        score = None
        cls = None
        if fmt.probs_start is not None and fmt.probs_len is not None:
            ps = v[fmt.probs_start : fmt.probs_start + fmt.probs_len]
            probs = np.asarray(ps, dtype=np.float32)
            if probs.size:
                cls = int(np.argmax(probs))
                score = float(probs[cls])

        # bbox slice
        bs = fmt.bbox_start
        be = bs + fmt.bbox_len
        if be > len(v):
            raise ValueError(f"Prediction too short for bbox slice [{bs}:{be}] (len={len(v)})")

        bbox_raw = v[bs:be]
        x1, y1, x2, y2 = _bbox_to_xyxy(bbox_raw, fmt.bbox_spec, img_w, img_h)

        # scalar indices override (if provided)
        if fmt.score_index is not None and fmt.score_index < len(v) and v[fmt.score_index] is not None:
            score = float(v[fmt.score_index])
        if fmt.cls_index is not None and fmt.cls_index < len(v) and v[fmt.cls_index] is not None:
            cls = int(v[fmt.cls_index])

        label = None
        if fmt.label_index is not None and fmt.label_index < len(v) and v[fmt.label_index] is not None:
            label = str(v[fmt.label_index])

        if (not label) and class_names is not None and cls is not None and 0 <= cls < len(class_names):
            label = class_names[cls]

        return (x1, y1, x2, y2), score, cls, label

    raise TypeError("Unsupported prediction item type for the chosen format.")


# ============================================================
# Annotator
# ============================================================

class Annotator:
    """
    Configure once in __init__ (image options + annotation options + drawing preset),
    then call .draw(image, preds) repeatedly.
    """

    def __init__(
        self,
        *,
        image_opts: ImageOptions = ImageOptions(),
        ann_opts: AnnotationOptions = AnnotationOptions(),
        preset: str = "classic",
        preset_overrides: Optional[Mapping[str, Any]] = None,
        font_path: Optional[str] = None,
        color_fn: Optional[Callable[[Optional[int]], Tuple[int, int, int]]] = None,
        output_dir: Optional[str | Path] = None,
        auto_mkdir: bool = True, 
    ):
        self.image_opts = image_opts
        self.ann_opts = ann_opts

        base = dict(PRESETS.get(preset, PRESETS["classic"]))
        if preset_overrides:
            base.update(dict(preset_overrides))
        self.style = base

        self.font_path = font_path
        self.color_fn = color_fn

        self.output_dir: Optional[Path] = Path(output_dir) if output_dir else None
        if self.output_dir and auto_mkdir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def draw(
        self,
        image: Any,
        preds: Iterable[Any],
        *,
        overrides: Optional[Mapping[str, Any]] = None,

        score_threshold: Optional[float] = None,  # e.g. 0.1

        # NEW: optional GT / targets to draw (same format as preds by default)
        targets: Optional[Iterable[Any]] = None,
        target_format: Optional[Any] = None,  # if None -> uses self.ann_opts.format
        target_overrides: Optional[Mapping[str, Any]] = None,

        # NEW: optional NMS over preds before drawing
        nms: bool = False,
        nms_iou: float = 0.5,
        nms_score: Optional[float] = None,   # e.g. 0.5
        nms_class_aware: bool = True,
        nms_topk: Optional[int] = None,      # e.g. 100
    ) -> Any:
        arr_in, kind, meta_kind = _to_numpy(image)
        arr_u8, meta_img = _ensure_uint8_hwc(arr_in, self.image_opts, kind)

        pil_img = Image.fromarray(arr_u8)
        draw = ImageDraw.Draw(pil_img, "RGBA")

        style = dict(self.style)
        if overrides:
            style.update(dict(overrides))

        font = _load_font(self.font_path, int(style.get("font_size", 16)))

        w, h = pil_img.size
        class_names = self.ann_opts.class_names
        fmt = self.ann_opts.format

        # ------------------------------------------------------------
        # 1) Parse all preds once (needed for optional NMS)
        # ------------------------------------------------------------
        parsed_preds: List[Tuple[Tuple[float, float, float, float], Optional[float], Optional[int], Optional[str]]] = []
        for p in preds:
            parsed_preds.append(_parse_pred(p, fmt, w, h, class_names))

        # ------------------------------------------------------------
        # NEW: score threshold filtering (before NMS & drawing)
        # ------------------------------------------------------------
        if score_threshold is not None:
            parsed_preds = [
                p for p in parsed_preds
                if p[1] is not None and float(p[1]) >= float(score_threshold)
            ]

        if nms:
            parsed_preds = _apply_nms_to_parsed(
                parsed_preds,
                iou_threshold=float(nms_iou),
                score_threshold=nms_score,
                class_aware=bool(nms_class_aware),
                topk=nms_topk,
            )

        # ------------------------------------------------------------
        # 2) Draw predictions (same drawing code as before)
        # ------------------------------------------------------------
        for (x1, y1, x2, y2), score, cls, label in parsed_preds:
            # sort + clamp
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            x1 = float(np.clip(x1, 0, w - 1))
            y1 = float(np.clip(y1, 0, h - 1))
            x2 = float(np.clip(x2, 0, w - 1))
            y2 = float(np.clip(y2, 0, h - 1))

            # color
            if style.get("color") is not None:
                rgb = tuple(style["color"])
            elif self.color_fn is not None:
                rgb = tuple(self.color_fn(cls))
            else:
                rgb = _color_for_class(cls, scheme=str(style.get("color_scheme", "hash")))

            thickness = int(style.get("thickness", 3))

            # optional filled box
            if style.get("box_fill"):
                alpha = int(style.get("box_fill_alpha", 60))
                draw.rectangle([x1, y1, x2, y2], fill=(rgb[0], rgb[1], rgb[2], alpha))

            # outline (rect or corners)
            if style.get("box", True):
                if style.get("box_style", "rect") == "corners":
                    frac = float(style.get("corner_frac", 0.22))
                    lx = max(1.0, (x2 - x1) * frac)
                    ly = max(1.0, (y2 - y1) * frac)
                    # TL
                    draw.line([(x1, y1), (x1 + lx, y1)], fill=rgb, width=thickness)
                    draw.line([(x1, y1), (x1, y1 + ly)], fill=rgb, width=thickness)
                    # TR
                    draw.line([(x2 - lx, y1), (x2, y1)], fill=rgb, width=thickness)
                    draw.line([(x2, y1), (x2, y1 + ly)], fill=rgb, width=thickness)
                    # BL
                    draw.line([(x1, y2 - ly), (x1, y2)], fill=rgb, width=thickness)
                    draw.line([(x1, y2), (x1 + lx, y2)], fill=rgb, width=thickness)
                    # BR
                    draw.line([(x2, y2 - ly), (x2, y2)], fill=rgb, width=thickness)
                    draw.line([(x2 - lx, y2), (x2, y2)], fill=rgb, width=thickness)
                else:
                    for t in range(thickness):
                        draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=rgb)

            # label text build
            if style.get("label", True):
                parts: List[str] = []

                if style.get("show_class") and cls is not None:
                    parts.append(str(cls))
                if style.get("show_label") and label:
                    parts.append(str(label))
                if style.get("show_score") and score is not None:
                    fmt_s = str(style.get("score_fmt", "{:.2f}"))
                    try:
                        parts.append(fmt_s.format(score))
                    except Exception:
                        parts.append(f"{float(score):.2f}")

                text = str(style.get("sep", " ")).join([p for p in parts if p])
                if text:
                    pad = int(style.get("text_padding", 4))
                    tb = draw.textbbox((0, 0), text, font=font)
                    tw, th = (tb[2] - tb[0]), (tb[3] - tb[1])

                    pos = str(style.get("label_position", "tl")).lower()
                    if pos == "tl":
                        tx, ty = x1, y1 - th - 2 * pad
                        if ty < 0:
                            ty = y1
                    elif pos == "tr":
                        tx, ty = x2 - tw - 2 * pad, y1 - th - 2 * pad
                        if ty < 0:
                            ty = y1
                    elif pos == "bl":
                        tx, ty = x1, y2
                        if ty + th + 2 * pad > h:
                            ty = max(0, y2 - th - 2 * pad)
                    else:  # br
                        tx, ty = x2 - tw - 2 * pad, y2
                        if ty + th + 2 * pad > h:
                            ty = max(0, y2 - th - 2 * pad)

                    tx = float(np.clip(tx, 0, max(0, w - (tw + 2 * pad))))
                    ty = float(np.clip(ty, 0, max(0, h - (th + 2 * pad))))

                    if style.get("label_bg", True):
                        a = int(style.get("label_bg_alpha", 220))
                        draw.rectangle(
                            [tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad],
                            fill=(rgb[0], rgb[1], rgb[2], a),
                        )
                        text_fill = (0, 0, 0, 255)
                    else:
                        text_fill = (rgb[0], rgb[1], rgb[2], 255)

                    if style.get("text_shadow"):
                        spx = int(style.get("text_shadow_px", 2))
                        draw.text((tx + pad + spx, ty + pad + spx), text, font=font, fill=(0, 0, 0, 180))

                    draw.text((tx + pad, ty + pad), text, font=font, fill=text_fill)

        # ------------------------------------------------------------
        # 3) Draw targets/GT last (on top), with independent overrides
        # ------------------------------------------------------------
        if targets is not None:
            gt_style = dict(style)
            # sensible GT defaults (you can override)
            gt_style.setdefault("color", (255, 255, 255))          # white boxes
            gt_style.setdefault("box_fill", False)
            gt_style.setdefault("thickness", max(2, int(style.get("thickness", 3))))
            gt_style.setdefault("show_score", False)               # GT typically has no score
            gt_style.setdefault("label_bg", True)
            gt_style.setdefault("label_bg_alpha", 180)

            if target_overrides:
                gt_style.update(dict(target_overrides))

            gt_fmt = target_format if target_format is not None else fmt

            parsed_gt: List[Tuple[Tuple[float, float, float, float], Optional[float], Optional[int], Optional[str]]] = []
            for t in targets:
                parsed_gt.append(_parse_pred(t, gt_fmt, w, h, class_names))

            # draw GT with gt_style (reuse the same drawing logic, but using gt_style + font)
            for (x1, y1, x2, y2), score, cls, label in parsed_gt:
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                x1 = float(np.clip(x1, 0, w - 1))
                y1 = float(np.clip(y1, 0, h - 1))
                x2 = float(np.clip(x2, 0, w - 1))
                y2 = float(np.clip(y2, 0, h - 1))

                rgb = tuple(gt_style["color"]) if gt_style.get("color") is not None else (255, 255, 255)
                thickness = int(gt_style.get("thickness", 3))

                # outline only for GT by default
                if gt_style.get("box", True):
                    if gt_style.get("box_style", "rect") == "corners":
                        frac = float(gt_style.get("corner_frac", 0.22))
                        lx = max(1.0, (x2 - x1) * frac)
                        ly = max(1.0, (y2 - y1) * frac)
                        draw.line([(x1, y1), (x1 + lx, y1)], fill=rgb, width=thickness)
                        draw.line([(x1, y1), (x1, y1 + ly)], fill=rgb, width=thickness)
                        draw.line([(x2 - lx, y1), (x2, y1)], fill=rgb, width=thickness)
                        draw.line([(x2, y1), (x2, y1 + ly)], fill=rgb, width=thickness)
                        draw.line([(x1, y2 - ly), (x1, y2)], fill=rgb, width=thickness)
                        draw.line([(x1, y2), (x1 + lx, y2)], fill=rgb, width=thickness)
                        draw.line([(x2, y2 - ly), (x2, y2)], fill=rgb, width=thickness)
                        draw.line([(x2 - lx, y2), (x2, y2)], fill=rgb, width=thickness)
                    else:
                        for t_ in range(thickness):
                            draw.rectangle([x1 - t_, y1 - t_, x2 + t_, y2 + t_], outline=rgb)

                # GT label (optional)
                if gt_style.get("label", True):
                    parts: List[str] = []
                    if gt_style.get("show_class") and cls is not None:
                        parts.append(str(cls))
                    if gt_style.get("show_label") and label:
                        parts.append(str(label))
                    # GT score typically off; but allow if someone passed it
                    if gt_style.get("show_score") and score is not None:
                        fmt_s = str(gt_style.get("score_fmt", "{:.2f}"))
                        try:
                            parts.append(fmt_s.format(score))
                        except Exception:
                            parts.append(f"{float(score):.2f}")

                    text = str(gt_style.get("sep", " ")).join([p for p in parts if p])
                    if text:
                        pad = int(gt_style.get("text_padding", 4))
                        tb = draw.textbbox((0, 0), text, font=font)
                        tw, th = (tb[2] - tb[0]), (tb[3] - tb[1])

                        tx, ty = x1, y1 - th - 2 * pad
                        if ty < 0:
                            ty = y1
                        tx = float(np.clip(tx, 0, max(0, w - (tw + 2 * pad))))
                        ty = float(np.clip(ty, 0, max(0, h - (th + 2 * pad))))

                        if gt_style.get("label_bg", True):
                            a = int(gt_style.get("label_bg_alpha", 180))
                            draw.rectangle([tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad],
                                        fill=(rgb[0], rgb[1], rgb[2], a))
                            text_fill = (0, 0, 0, 255)
                        else:
                            text_fill = (rgb[0], rgb[1], rgb[2], 255)

                        draw.text((tx + pad, ty + pad), text, font=font, fill=text_fill)

        out_u8 = np.array(pil_img)
        out_arr = _restore_from_uint8_hwc(out_u8, meta_img)
        return _from_numpy(out_arr, kind, meta_kind, self.image_opts)

    
    def save(self,image,*,name: str, ext: Optional[str] = None,) -> Path:
        """
        Save an image to the annotator's output directory.

        Args:
            image: PIL / numpy / torch image
            name: filename without extension (e.g. "frame_001")
            ext: optional extension (".png", ".jpg"); inferred if None

        Returns:
            Path to saved file
        """
        if self.output_dir is None:
            raise ValueError("Annotator.output_dir is not set")

        # Convert to PIL safely
        if isinstance(image, Image.Image):
            pil_img = image
        else:
            arr, _, _ = _to_numpy(image)
            if arr.ndim == 3 and arr.shape[2] == 3:
                pil_img = Image.fromarray(arr.astype(np.uint8))
            elif arr.ndim == 2:
                pil_img = Image.fromarray(arr.astype(np.uint8), mode="L")
            else:
                raise ValueError("Unsupported image shape for saving")

        # Infer extension
        if ext is None:
            ext = ".png"
        if not ext.startswith("."):
            ext = "." + ext

        path = self.output_dir / f"{name}{ext}"
        pil_img.save(path)

        return path



def tv_det_to_xyxy_score_cls(results, *, device="cpu"):
    """
    Convert torchvision detection output to:
    [x1, y1, x2, y2, score, class_id]

    Args:
        results: list with one dict containing 'boxes', 'scores', 'labels'
        device: 'cpu' or 'cuda'

    Returns:
        List[List[float|int]]
    """
    res = results[0]

    boxes = res["boxes"].to(device)
    if res.get("scores") is not None:
        scores = res["scores"].to(device)
    else:
        scores = torch.ones((boxes.shape[0],), device=device)
    labels = res["labels"].to(device)

    return [
        [*box.tolist(), float(score), int(cls)]
        for box, score, cls in zip(boxes, scores, labels)
    ]

def target_to_xyxy_score_cls(target, *, device="cpu"):
    """
    Convert a single target dict to:
    [x1, y1, x2, y2, score, class_id]

    Args:
        target: dict with 'boxes', optional 'scores', 'labels'
        device: 'cpu' or 'cuda'

    Returns:
        List[List[float|int]]
    """
    target = target[0]

    boxes = target["boxes"].to(device)
    if target.get("scores") is not None:
        scores = target["scores"].to(device)
    else:
        scores = torch.ones((boxes.shape[0],), device=device)
    labels = target["labels"].to(device)

    # Scale normalized boxes [0,1] to pixel coordinates using orig_size
    orig_size = target.get("size")
    if orig_size is not None:
        orig_size = torch.as_tensor(orig_size, device=device, dtype=boxes.dtype)
        # Assume boxes are [x, y, w, h] normalized
        # Scale to [width, height, width, height]
        scale = torch.tensor([orig_size[1], orig_size[0], orig_size[1], orig_size[0]], device=device, dtype=boxes.dtype)
        boxes = boxes * scale
        # Convert from [x, y, w, h] to [x1, y1, x2, y2]
        boxes[:, 2] += boxes[:, 0]  # x2 = x + w
        boxes[:, 3] += boxes[:, 1]  # y2 = y + h

    return [
        [*box.tolist(), float(score), int(cls)]
        for box, score, cls in zip(boxes, scores, labels)
    ]

def target_to_xyxy_score_cls_single(target, *, device="cpu"):
    """
    Convert a single target dict to:
    [x1, y1, x2, y2, score, class_id]

    Args:
        target: dict with 'boxes', optional 'scores', 'labels'
        device: 'cpu' or 'cuda'

    Returns:
        List[List[float|int]]
    """

    boxes = target["boxes"].to(device)
    if target.get("scores") is not None:
        scores = target["scores"].to(device)
    else:
        scores = torch.ones((boxes.shape[0],), device=device)
    labels = target["labels"].to(device)

    return [
        [*box.tolist(), float(score), int(cls)]
        for box, score, cls in zip(boxes, scores, labels)
    ]


def unwrap_tensor(x: Any) -> torch.Tensor:
    """
    Convert common wrapper types to a plain torch.Tensor.

    Supports:
      - torch.Tensor
      - NestedTensor-like objects with .tensors
      - (tensor, mask) tuples/lists (takes first element)
      - dicts with common keys
    """
    if torch.is_tensor(x):
        return x

    # DETR-style NestedTensor: has .tensors (and usually .mask)
    if hasattr(x, "tensors") and torch.is_tensor(getattr(x, "tensors")):
        return getattr(x, "tensors")

    # Sometimes: NestedTensor.decompose() -> (tensor, mask)
    if hasattr(x, "decompose"):
        t, *_ = x.decompose()
        if torch.is_tensor(t):
            return t

    # tuple/list wrapper
    if isinstance(x, (tuple, list)) and x:
        return unwrap_tensor(x[0])

    # dict wrapper
    if isinstance(x, dict):
        for k in ("image", "images", "tensor", "tensors", "x"):
            if k in x:
                return unwrap_tensor(x[k])

    raise TypeError(f"Unsupported tensor wrapper type: {type(x)!r}")



def tensor_to_uint8_hwc(
    x: torch.Tensor,
    *,
    mean: Optional[Sequence[float]] = (0.485, 0.456, 0.406),
    std: Optional[Sequence[float]] = (0.229, 0.224, 0.225),
    assume_range_0_1_if_no_meanstd: bool = True,
    clamp: bool = True,
) -> np.ndarray:
    """
    Convert a torch image tensor to uint8 HWC (RGB).

    Supports:
      - CHW or NCHW
      - on GPU or CPU
      - normalized via (x - mean) / std (common ImageNet)
      - or raw floats in [0,1] if mean/std is None

    Returns:
      uint8 array of shape (H,W,3) if input was CHW
      or (N,H,W,3) if input was NCHW
    """

    if x.ndim not in (3, 4):
        raise ValueError(f"Expected CHW or NCHW tensor, got shape {tuple(x.shape)}")

    t = x.detach()

    # If batch, operate per batch
    if t.ndim == 3:
        t = t.unsqueeze(0)  # -> NCHW

    # Ensure float for math
    if not torch.is_floating_point(t):
        t = t.float()

    # Denormalize if mean/std provided
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, device=t.device, dtype=t.dtype).view(1, -1, 1, 1)
        std_t = torch.tensor(std, device=t.device, dtype=t.dtype).view(1, -1, 1, 1)
        t = t * std_t + mean_t
    else:
        # Otherwise assume it might already be in [0,1] floats (common)
        if assume_range_0_1_if_no_meanstd:
            pass  # leave as-is

    if clamp:
        t = t.clamp(0.0, 1.0)

    # NCHW -> NHWC
    t = t.permute(0, 2, 3, 1).contiguous()

    # to uint8
    arr = (t * 255.0).round().to(torch.uint8).cpu().numpy()

    # If original was CHW, return single image HWC
    return arr[0] if x.ndim == 3 else arr


def tensor_to_pil(
    x: torch.Tensor,
    *,
    mean: Optional[Sequence[float]] = (0.485, 0.456, 0.406),
    std: Optional[Sequence[float]] = (0.229, 0.224, 0.225),
) -> Union[Image.Image, list[Image.Image]]:
    x = unwrap_tensor(x)
    arr = tensor_to_uint8_hwc(x, mean=mean, std=std)
    if arr.ndim == 3:
        return Image.fromarray(arr, mode="RGB")
    return [Image.fromarray(arr[i], mode="RGB") for i in range(arr.shape[0])]


# Add these helpers somewhere in your annotator.py (near other helpers)

from typing import Iterable, Any, Optional, Mapping, List, Tuple
import numpy as np


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (N,4) xyxy, b: (M,4) xyxy -> IoU (N,M)
    """
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)

    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-9)


def _nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    iou_threshold: float = 0.5,
    topk: Optional[int] = None,
) -> List[int]:
    """
    Pure numpy NMS.
    boxes: (N,4) float xyxy
    scores: (N,) float
    returns: kept indices (in original order of boxes/scores)
    """
    if boxes.size == 0:
        return []

    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if topk is not None and len(keep) >= topk:
            break

        if order.size == 1:
            break

        rest = order[1:]
        ious = _iou_xyxy(boxes[i:i + 1], boxes[rest]).reshape(-1)
        rest = rest[ious <= iou_threshold]
        order = rest

    return keep

def gt_target_to_xyxy_pixels(target: dict) -> list:
    """
    Converts DETR-style target dict with:
      target["boxes"] in normalized cxcywh (0..1)
      target["labels"] as class ids
      target["size"] = tensor([H,W])
    into annotator format:
      [x1, y1, x2, y2, score, cls]
    """
    boxes = target["boxes"]
    labels = target.get("labels", None)
    if labels is None:
        raise KeyError("target must contain 'labels' for class ids")

    # size is [H, W]
    h, w = target["size"].tolist()

    # boxes: (N,4) cxcywh normalized -> scale to pixels
    boxes_px = boxes.clone()
    boxes_px[:, 0] *= w  # cx
    boxes_px[:, 2] *= w  # bw
    boxes_px[:, 1] *= h  # cy
    boxes_px[:, 3] *= h  # bh

    # cxcywh -> xyxy
    cx, cy, bw, bh = boxes_px.unbind(-1)
    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0

    xyxy = torch.stack([x1, y1, x2, y2], dim=-1).detach().cpu()
    labels = labels.detach().cpu()

    # GT has no real score; use 1.0 so your score_threshold doesn't drop it
    return [[*b.tolist(), 1.0, int(c)] for b, c in zip(xyxy, labels)]


def _apply_nms_to_parsed(
    parsed: List[Tuple[Tuple[float, float, float, float], Optional[float], Optional[int], Optional[str]]],
    *,
    iou_threshold: float,
    score_threshold: Optional[float],
    class_aware: bool,
    topk: Optional[int],
) -> List[Tuple[Tuple[float, float, float, float], Optional[float], Optional[int], Optional[str]]]:
    """
    parsed: list of (xyxy, score, cls, label)
    """
    if not parsed:
        return parsed

    # score filter first
    if score_threshold is not None:
        parsed = [p for p in parsed if (p[1] is not None and float(p[1]) >= score_threshold)]
        if not parsed:
            return parsed

    boxes = np.asarray([p[0] for p in parsed], dtype=np.float32)
    scores = np.asarray([float(p[1]) if p[1] is not None else 0.0 for p in parsed], dtype=np.float32)
    clses = np.asarray([(-1 if p[2] is None else int(p[2])) for p in parsed], dtype=np.int32)

    if not class_aware:
        keep = _nms_xyxy(boxes, scores, iou_threshold=iou_threshold, topk=topk)
        return [parsed[i] for i in keep]

    # class-aware NMS (run per class id)
    kept_all: List[int] = []
    for c in np.unique(clses):
        idxs = np.where(clses == c)[0]
        if idxs.size == 0:
            continue
        keep_local = _nms_xyxy(
            boxes[idxs],
            scores[idxs],
            iou_threshold=iou_threshold,
            topk=topk,  # topk per class; set None if you want global only
        )
        kept_all.extend(idxs[keep_local].tolist())

    # If a global topk is desired, sort by score after merging
    kept_all = sorted(set(kept_all), key=lambda i: scores[i], reverse=True)
    if topk is not None:
        kept_all = kept_all[:topk]
    return [parsed[i] for i in kept_all]




# ============================================================
# Examples: configuring formats quickly
# ============================================================

# if __name__ == "__main__":
#     # Example 1: preds are [x1,y1,x2,y2, conf, cls] in pixels (xyxy)
#     ann = Annotator(
#         image_opts=ImageOptions(channel_order="rgb"),
#         ann_opts=AnnotationOptions(
#             format=PredFormat(
#                 bbox_start=0,
#                 bbox_len=4,
#                 bbox_spec=BBoxSpec(fmt="xyxy", normalized=False),
#                 score_index=4,
#                 cls_index=5,
#             ),
#             class_names=["a", "b", "c"],
#         ),
#         preset="filled",
#     )

#     img = Image.new("RGB", (640, 360), (25, 25, 25))
#     preds = [
#         [50, 60, 220, 200, 0.93, 1],
#         [300, 80, 520, 250, 0.81, 2],
#     ]
#     out = ann.draw(img, preds)
#     out.save("demo_xyxy.png")

#     # Example 2: preds are [cls, conf, cx, cy, w, h] normalized cxcywh (common custom)
#     ann2 = Annotator(
#         ann_opts=AnnotationOptions(
#             format=PredFormat(
#                 bbox_start=2,
#                 bbox_len=4,
#                 bbox_spec=BBoxSpec(fmt="cxcywh", normalized=True),
#                 score_index=1,
#                 cls_index=0,
#             )
#         ),
#         preset="classic",
#     )
#     preds2 = [[1, 0.7, 0.5, 0.5, 0.3, 0.4]]
#     out2 = ann2.draw(img, preds2)
#     out2.save("demo_cxcywh_norm.png")

#     # Example 3: dict format
#     ann3 = Annotator(
#         ann_opts=AnnotationOptions(
#             format=DictPredFormat(
#                 bbox_keys=("bbox",),
#                 score_keys=("conf",),
#                 cls_keys=("cls",),
#                 label_keys=("name",),
#                 bbox_spec=BBoxSpec(fmt="xywh", normalized=False),
#             )
#         ),
#         preset="classic",
#     )
#     preds3 = [{"bbox": [30, 40, 100, 120], "conf": 0.88, "cls": 0, "name": "thing"}]
#     out3 = ann3.draw(img, preds3)
#     out3.save("demo_dict.png")
