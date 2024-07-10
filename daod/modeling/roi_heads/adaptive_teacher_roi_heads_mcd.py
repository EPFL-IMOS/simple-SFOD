# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, Instances

from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from daod.modeling.roi_heads import AdaptiveTeacherStandardROIHeads


@ROI_HEADS_REGISTRY.register()
class AdaptiveTeacherStandardROIHeadsWithMCD(AdaptiveTeacherStandardROIHeads):

    def __init__(self, cfg, input_shape):
        super(AdaptiveTeacherStandardROIHeads, self).__init__(cfg, input_shape)
        self.mcd_samples = cfg.MODEL.ROI_HEADS.MCD_SAMPLES

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        
        if (
            self.training and compute_loss
        ) or compute_val_loss:  # apply if training loss or val loss
            box_features = self.box_head(box_features)
            predictions = self.box_predictor(box_features)
            losses = self.box_predictor.losses(predictions, proposals)

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions, box_features
        else:
            # Perform Monte-Carlo Dropout inferences
            box_features, predictions = [], []
            for n in range(self.mcd_samples):
                box_features.append(self.box_head(box_features))
                predictions.append(self.box_predictor(box_features))
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions
