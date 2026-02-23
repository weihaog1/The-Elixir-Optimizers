"""Custom NMS and plot utilities with belonging support.

Ported from KataCR's katacr/yolov8/custom_utils.py.
NMS outputs 7-column detections: (x1, y1, x2, y2, conf, cls, belonging).
"""

import contextlib

from ultralytics.utils.plotting import (
    threaded, np, torch, math, cv2, Annotator, Path, ops, colors,
)


@threaded
def plot_images(
    images,
    batch_idx,
    cls,
    bboxes=np.zeros(0, dtype=np.float32),
    confs=None,
    masks=np.zeros(0, dtype=np.uint8),
    kpts=np.zeros((0, 51), dtype=np.float32),
    paths=None,
    fname="images.jpg",
    names=None,
    on_plot=None,
    max_subplots=16,
    save=True,
    conf_thres=0.25,
):
    """Plot image grid with labels including belonging annotation."""
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(cls, torch.Tensor):
        cls = cls.cpu().numpy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.cpu().numpy()
    if isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.cpu().numpy()

    max_size = 1920 * 4
    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)
    ns = np.ceil(bs**0.5)
    if np.max(images[0]) <= 1:
        images *= 255

    # Build mosaic
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))
        mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0)

    # Resize
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)
    annotator = Annotator(
        mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names
    )
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)
        if paths and paths[i]:
            annotator.text(
                (x + 5, y + 5),
                text=Path(paths[i]).name[:40],
                txt_color=(220, 220, 220),
            )
        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype("int")  # (N, 2): cls, bel
            labels = confs is None

            if len(bboxes):
                boxes = bboxes[idx]
                conf = confs[idx] if confs is not None else None
                is_obb = boxes.shape[-1] == 5
                boxes = (
                    ops.xywhr2xyxyxyxy(boxes) if is_obb else ops.xywh2xyxy(boxes)
                )
                if len(boxes):
                    if boxes[:, :4].max() <= 1.1:
                        boxes[..., 0::2] *= w
                        boxes[..., 1::2] *= h
                    elif scale < 1:
                        boxes[..., :4] *= scale
                boxes[..., 0::2] += x
                boxes[..., 1::2] += y
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j, 0]
                    bel = classes[j, 1]
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    if labels or conf[j] > conf_thres:
                        label = (
                            f"{c}{bel}" if labels else f"{c}{bel} {conf[j]:.1f}"
                        )
                        annotator.box_label(box, label, color=color)
    if not save:
        return np.asarray(annotator.im)
    annotator.im.save(fname)
    if on_plot:
        on_plot(fname)


import time

import torchvision
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import xywh2xyxy


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
):
    """NMS with belonging output.

    Returns list of tensors with 7 columns per detection:
    (x1, y1, x2, y2, confidence, class, belonging)
    """
    assert 0 <= conf_thres <= 1, (
        f"Invalid confidence threshold {conf_thres}"
    )
    assert 0 <= iou_thres <= 1, f"Invalid IoU threshold {iou_thres}"
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = prediction[:, 4:mi].amax(1) > conf_thres

    time_limit = 2.0 + max_time_img * bs
    multi_label &= nc > 1

    prediction = prediction.transpose(-1, -2)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])
        else:
            prediction = torch.cat(
                (xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1
            )

    t = time.time()
    output = [torch.zeros((0, 7 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        # Split: box(4), class_scores(nc-1), belonging(1), masks(nm)
        box, cls, bel, mask = x.split((4, nc - 1, 1, nm), 1)

        # Best class only
        conf, j = cls.max(1, keepdim=True)
        bel = (bel > 0.5).float()
        x = torch.cat((box, conf, j.float(), bel), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]
        boxes = x[:, :4] + c
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(
                f"WARNING: NMS time limit {time_limit:.3f}s exceeded"
            )
            break

    return output
