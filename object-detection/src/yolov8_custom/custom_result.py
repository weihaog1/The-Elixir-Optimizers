"""Custom Results and Boxes classes with belonging support.

Ported from KataCR's katacr/yolov8/custom_result.py.
CRBoxes handles 7-column data: (x1, y1, x2, y2, conf, cls, belonging)
or 8-column with tracking: (x1, y1, x2, y2, track_id, conf, cls, belonging).
"""

from ultralytics.engine.results import (
    Results, Boxes, ops, torch, Annotator, deepcopy, LetterBox,
    colors, Path, LOGGER, save_one_box, np,
)


class CRBoxes(Boxes):
    def __init__(self, boxes, orig_shape) -> None:
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in (7, 8), f"expected 7 or 8 values but got {n}"
        assert isinstance(boxes, (torch.Tensor, np.ndarray))
        self.data = boxes
        self.orig_shape = orig_shape
        self.is_track = n == 8

    @property
    def id(self):
        """Track IDs (if available)."""
        return self.data[:, -4] if self.is_track else None

    @property
    def cls(self):
        """Class and belonging: (N, 2) tensor."""
        return self.data[:, -2:]

    @property
    def conf(self):
        """Confidence scores."""
        return self.data[:, -3]


class CRResults(Results):
    def __init__(
        self,
        orig_img,
        path,
        names,
        boxes=None,
        logits_boxes=None,
        masks=None,
        probs=None,
        keypoints=None,
        obb=None,
    ) -> None:
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.orig_boxes = boxes
        self.logits_boxes = logits_boxes
        self.boxes = (
            CRBoxes(boxes, self.orig_shape) if boxes is not None else None
        )
        self.masks = None
        self.probs = None
        self.keypoints = None
        self.obb = None
        self.speed = {
            "preprocess": None,
            "inference": None,
            "postprocess": None,
        }
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"

    def update(self, boxes=None, masks=None, probs=None, obb=None):
        if boxes is not None:
            self.boxes = CRBoxes(
                ops.clip_boxes(boxes, self.orig_shape), self.orig_shape
            )

    def plot(
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        show=False,
        save=False,
        filename=None,
    ):
        """Plot detections with belonging annotation (0=ally, 1=enemy)."""
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (
                (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )

        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = (
            self.obb if is_obb else self.boxes,
            boxes,
        )
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),
            example=names,
        )

        if pred_boxes is not None and show_boxes:
            for d in reversed(pred_boxes):
                c = int(d.cls[0, 0])
                bel = int(d.cls[0, 1])
                c_conf = float(d.conf) if conf else None
                track_id = (
                    None if d.id is None else int(d.id.item())
                )
                bel_str = "A" if bel == 0 else "E"
                name = (
                    ("" if track_id is None else f"id:{track_id} ")
                    + names[c]
                    + f"({bel_str})"
                )
                label = (
                    (f"{name} {c_conf:.2f}" if c_conf else name)
                    if labels
                    else None
                )
                box = (
                    d.xyxyxyxy.reshape(-1, 4, 2).squeeze()
                    if is_obb
                    else d.xyxy.squeeze()
                )
                # Blue for ally, red for enemy
                color = (255, 100, 50) if bel == 0 else (50, 50, 255)
                annotator.box_label(
                    box, label, color=color, rotated=is_obb
                )

        if pred_probs is not None and show_probs:
            text = ",\n".join(
                f"{names[j] if names else j} {pred_probs.data[j]:.2f}"
                for j in pred_probs.top5
            )
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=(255, 255, 255))

        if show:
            annotator.show(self.path)
        if save:
            annotator.save(filename)

        return annotator.result()

    def get_data(self):
        if not isinstance(self.boxes.data, np.ndarray):
            if self.boxes.data.device != "cpu":
                self.boxes.data = self.boxes.data.cpu().numpy()
            else:
                self.boxes.data = self.boxes.data.numpy()
        return self.boxes.data

    def verbose(self):
        """Return log string for each task."""
        log_string = ""
        probs = self.probs
        boxes = self.boxes
        if len(self) == 0:
            return (
                log_string
                if probs is not None
                else f"{log_string}(no detections), "
            )
        if probs is not None:
            log_string += (
                f"{', '.join(f'{self.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "
            )
        if boxes:
            for c in boxes.cls[:, 0].unique():
                n = (boxes.cls[:, 0] == c).sum()
                log_string += (
                    f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
                )
        return log_string

    def save_txt(self, txt_file, save_conf=False):
        """Save predictions to txt with belonging."""
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        texts = []
        if boxes:
            for j, d in enumerate(boxes):
                c = int(d.cls[0, 0])
                bel = int(d.cls[0, 1])
                c_conf = float(d.conf)
                track_id = (
                    None if d.id is None else int(d.id.item())
                )
                line = (
                    c,
                    bel,
                    *(
                        d.xyxyxyxyn.view(-1)
                        if is_obb
                        else d.xywhn.view(-1)
                    ),
                )
                line += (c_conf,) * save_conf + (
                    () if track_id is None else (track_id,)
                )
                texts.append(("%g " * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)
            with open(txt_file, "a") as f:
                f.writelines(text + "\n" for text in texts)

    def tojson(self, normalize=False):
        """Convert detections to JSON with belonging."""
        if self.probs is not None:
            LOGGER.warning("Classify task does not support tojson yet.")
            return

        import json

        results = []
        data = self.boxes.data.cpu().tolist()
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):
            box = {
                "x1": row[0] / w,
                "y1": row[1] / h,
                "x2": row[2] / w,
                "y2": row[3] / h,
            }
            conf = row[-3]
            class_id = int(row[-2])
            bel = int(row[-1])
            name = self.names[class_id]
            result = {
                "name": name,
                "class": class_id,
                "confidence": conf,
                "box": box,
                "belong": bel,
            }
            if self.boxes.is_track:
                result["track_id"] = int(row[-4])
            results.append(result)

        return json.dumps(results, indent=2)
