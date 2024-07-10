from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import get_fed_loss_cls_weights
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

class SourceFreeFastRCNNOutputLayers(FastRCNNOutputLayers):
    def convert_bbox_scores(self, predictions, proposals):
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        result = self.fast_rcnn_inference_new(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            proposals
        )
        #print("result: ")
        #print(result)
        #print("predictions: ")
        #print(predictions)
        #print("proposals: ")
        #print(proposals)
        # result.gt_classes = proposals.gt_classes
        # result.gt_boxes = proposals.gt_boxes
        return result

    
    def fast_rcnn_inference_new(
        self,
        boxes: List[torch.Tensor],
        scores: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
        proposals
        ):
        """
        Call `fast_rcnn_inference_single_image` for all images.

        Args:
            boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
                boxes for each image. Element i has shape (Ri, K * 4) if doing
                class-specific regression, or (Ri, 4) if doing class-agnostic
                regression, where Ri is the number of predicted objects for image i.
                This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
            scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
            image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
            score_thresh (float): Only return detections with a confidence score exceeding this
                threshold.
            nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
            topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
                all detections.

        Returns:
            instances: (list[Instances]): A list of N instances, one for each image in the batch,
                that stores the topk most confidence detections.
            kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
                the corresponding boxes/scores index in [0, Ri) from the input, for image i.
        """
        result_per_image = [
            self.fast_rcnn_inference_single_image_new(
                boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, proposal
            )
            for scores_per_image, boxes_per_image, image_shape, proposal in zip(scores, boxes, image_shapes, proposals)
        ]
        return [x[0] for x in result_per_image], [x[1] for x in result_per_image]

    def fast_rcnn_inference_single_image_new(
        self,
        boxes,
        scores,
        image_shape: Tuple[int, int],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
        proposal
        ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Args:
            Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
            per image.

        Returns:
            Same as `fast_rcnn_inference`, but for only one image.
        """
        valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]

        scores = scores[:, :-1]
        num_bbox_reg_classes = boxes.shape[1] // 4
        # Convert to Boxes to use the `clip` function ...
        # print(len(boxes))
        boxes = Boxes(boxes.reshape(-1, 4))
        boxes.clip(image_shape)
        boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

        # 1. Filter results based on detection scores. It can make NMS more efficient
        #    by filtering out low-confidence detections.
        # print("score thresh")
        # print(score_thresh)
        # We do not filter out anything here
        filter_mask = scores > 0  # R x K
        # R' x 2. First column contains indices of the R predictions;
        # Second column contains indices of classes.
        filter_inds = filter_mask.nonzero()
        
        if num_bbox_reg_classes == 1:
            boxes = boxes[filter_inds[:, 0], 0]
        else:
            boxes = boxes[filter_mask]
        scores = scores[filter_mask]

        '''
        # 2. Apply NMS for each class independently.
        keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
        boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
        '''
        

        result = Instances(image_shape)
        result.pred_boxes = Boxes(boxes)
        result.scores = scores
        result.pred_classes = filter_inds[:, 1]
        #result.gt_classes = proposal.gt_classes
        #result.gt_boxes = proposal.gt_boxes     
        return result, filter_inds[:, 0]