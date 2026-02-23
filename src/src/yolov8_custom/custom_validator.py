"""Custom detection validator with belonging support.

Ported from KataCR's katacr/yolov8/custom_validator.py.
Validates on class only (belonging ignored for mAP computation).
Uses custom NMS that outputs 7-column detections.
"""

from ultralytics.models.yolo.detect.val import DetectionValidator, torch, Path
from ultralytics.utils import ops, colorstr

from src.yolov8_custom.custom_utils import plot_images, non_max_suppression
from src.generation.synthetic_dataset import CRDataset


def output_to_target(output, max_det=300):
    """Convert list-of-dicts output to target format for plotting.

    Input: list of dicts with keys bboxes, conf, cls, belonging
    Output: batch_id, cls_with_bel(2), xywh, conf
    """
    targets = []
    for i, o in enumerate(output):
        bboxes = o["bboxes"][:max_det].cpu()
        conf = o["conf"][:max_det].cpu().unsqueeze(1)
        cls = o["cls"][:max_det].cpu().unsqueeze(1)
        bel = o.get("extra", torch.zeros_like(cls))[:max_det].cpu()
        if bel.dim() == 1:
            bel = bel.unsqueeze(1)
        j = torch.full((conf.shape[0], 1), i)
        cls_bel = torch.cat([cls, bel[:, :1]], 1)  # (N, 2)
        targets.append(torch.cat((j, cls_bel, ops.xyxy2xywh(bboxes), conf), 1))
    targets = torch.cat(targets, 0).numpy()
    return targets[:, 0], targets[:, 1:3], targets[:, 3:-1], targets[:, -1]


class CRDetectionValidator(DetectionValidator):
    def plot_val_samples(self, batch, ni):
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"],
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def postprocess(self, preds):
        """Apply custom NMS with belonging."""
        raw = non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            multi_label=False,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
        )
        # Convert 7-column tensors to dict format
        results = []
        for x in raw:
            results.append({
                "bboxes": x[:, :4],
                "conf": x[:, 4],
                "cls": x[:, 5],
                "extra": x[:, 6:],  # belonging
            })
        return results

    def _prepare_batch(self, si, batch):
        """Prepare batch using class only (ignoring belonging) for mAP."""
        idx = batch["batch_idx"] == si
        cls_data = batch["cls"][idx]
        # Handle 2-column cls (class, belonging) - use class only for mAP
        if cls_data.dim() == 2 and cls_data.shape[1] >= 2:
            cls = cls_data[:, 0]
        else:
            cls = cls_data.squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if cls.shape[0]:
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(
                imgsz, device=self.device
            )[[1, 0, 1, 0]]
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

    def build_dataset(self, img_path, mode="val", batch=None):
        return CRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            cache=self.args.cache,
            augment=False,
            prefix=colorstr(f"{mode}: "),
            rect=True,
            batch_size=batch,
            stride=32,
            pad=0.0,
            single_cls=False,
            classes=None,
            fraction=1.0,
            data=self.data,
            seed=self.args.seed,
        )
