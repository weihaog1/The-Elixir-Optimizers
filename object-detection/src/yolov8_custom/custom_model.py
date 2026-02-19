"""Custom YOLOv8 detection model with belonging prediction.

Ported from KataCR's katacr/yolov8/custom_model.py.
Uses padding_belong as the last class channel to predict ally/enemy.
The loss treats cls as 2-column (class_id, belonging) throughout.
"""

from ultralytics.nn.tasks import DetectionModel, v8DetectionLoss
from ultralytics.utils.loss import torch, make_anchors, xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner


class CRDetectionModel(DetectionModel):
    def init_criterion(self):
        return CRDetectionLoss(self)


class CRDetectionLoss(v8DetectionLoss):

    def __init__(self, model):
        super().__init__(model)
        self.assigner = CRTaskAlignedAssigner(
            topk=10, num_classes=self.nc, alpha=0.5, beta=6.0
        )

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess targets with belonging: (batch_idx, cls, bel, x, y, w, h)."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]  # cls, bel, xywh
            out[..., 2:6] = xywh2xyxy(out[..., 2:6].mul_(scale_tensor))
        return out

    def __call__(self, preds, batch):
        """Calculate box, cls, and dfl loss with belonging support."""
        preds = self.parse_output(preds)
        return self.loss(preds, batch)

    def loss(self, preds, batch):
        """Calculate belonging-aware detection loss."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )

        # Targets: (N, 1+2+4) = (batch_idx, cls, bel, x, y, w, h)
        targets = torch.cat(
            (
                batch["batch_idx"].view(-1, 1),
                batch["cls"].view(-1, 2),
                batch["bboxes"],
            ),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device),
            batch_size,
            scale_tensor=imgsz[[1, 0, 1, 0]],
        )
        gt_labels, gt_bboxes = targets.split((2, 4), 2)  # (cls, bel), xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Predicted boxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss (includes belonging in last channel)
        loss[1] = (
            self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        )

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss * batch_size, loss.detach()


class CRTaskAlignedAssigner(TaskAlignedAssigner):

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """Compute targets with belonging in last channel of target_scores.

        Args:
            gt_labels: (b, max_num_obj, 2) - class_id and belonging
            gt_bboxes: (b, max_num_obj, 4)
            target_gt_idx: (b, h*w) - assigned GT indices
            fg_mask: (b, h*w) - foreground mask

        Returns:
            target_labels: (b, h*w, 2)
            target_bboxes: (b, h*w, 4)
            target_scores: (b, h*w, num_classes) with belonging in last channel
        """
        batch_ind = torch.arange(
            end=self.bs, dtype=torch.int64, device=gt_labels.device
        )[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes

        # (b, h*w, 2) - class_id and belonging per anchor
        target_labels = gt_labels.long().view(-1, gt_labels.shape[-1])[target_gt_idx]

        # (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        target_labels.clamp_(0)

        # Build target scores: one-hot for class, raw value for belonging
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )
        # One-hot encode class_id (first column of gt_labels)
        target_scores.scatter_(2, target_labels[..., 0:1], 1)
        # Set belonging in last channel
        target_scores[..., -1] = target_labels[..., 1]

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric using class scores only (not belonging)."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros(
            [self.bs, self.n_max_boxes, na],
            dtype=pd_bboxes.dtype,
            device=pd_bboxes.device,
        )
        bbox_scores = torch.zeros(
            [self.bs, self.n_max_boxes, na],
            dtype=pd_scores.dtype,
            device=pd_scores.device,
        )

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = (
            torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        )
        ind[1] = gt_labels[..., 0]  # class_id only (not belonging)

        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[
            mask_gt
        ]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps
