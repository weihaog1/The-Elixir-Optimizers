"""Custom detection predictor with belonging support.

Ported from KataCR's katacr/yolov8/custom_predict.py.
Uses custom NMS and returns CRResults with belonging.
"""

from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils import ops

from src.yolov8_custom.custom_utils import non_max_suppression
from src.yolov8_custom.custom_result import CRResults


class CRDetectionPredictor(BasePredictor):

    def postprocess(self, preds, img, orig_imgs):
        """Post-process predictions with belonging-aware NMS."""
        preds = non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            logits_pred = pred.clone()
            pred[:, :4] = ops.scale_boxes(
                img.shape[2:], pred[:, :4], orig_img.shape
            )
            img_path = self.batch[0][i]
            results.append(
                CRResults(
                    orig_img,
                    path=img_path,
                    names=self.model.names,
                    boxes=pred,
                    logits_boxes=logits_pred,
                )
            )
        return results
