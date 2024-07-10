"""
Conditional DA StandardROIHeads.
"""
from typing import Dict, List, Optional, Tuple
import torch

from detectron2.structures import Boxes, ImageList, Instances
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY

from daod.modeling.roi_heads import DAStandardROIHeads


@ROI_HEADS_REGISTRY.register()
class CDAStandardROIHeads(DAStandardROIHeads):
    """Conditional version of DAStandardROIHeads also returning box category scores."""
    @torch.no_grad()
    def sample_unlabeled_proposals(
        self, proposals: List[Instances],
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """

        proposals_sampled = []

        for proposals_per_image in proposals:
            sampled_idxs = torch.randperm(len(proposals_per_image))[:self.batch_size_per_image]
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_sampled.append(proposals_per_image)

        return proposals_sampled

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            if targets is not None:  # labeled (source)
                proposals = self.label_and_sample_proposals(proposals, targets)
            else:  # unlabeled (target)
                proposals = self.sample_unlabeled_proposals(proposals)

        if self.training:
            losses, box_features, box_scores = self._forward_box(features, proposals, targets)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses, box_features, box_scores
        else:
            pred_instances = self._forward_box(features, proposals, targets)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances], targets: Optional[List[Instances]] = None):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a tuple with a dict of losses and box features.
            In inference, a tuple with a list of predicted `Instances` and box features.
        """
        # print("BOX FEATURES DEBUG START")
        # print({f: features[f][0].size() for f in self.box_in_features})
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        # print(box_features.size())
        box_features = self.box_head(box_features)
        # print(box_features.size())
        # print("BOX FEATURES DEBUG END")
        predictions = self.box_predictor(box_features)  # scores, proposal_deltas 

        if self.training:
            if targets is not None:
                losses = self.box_predictor.losses(predictions, proposals)
                # proposals is modified in-place below, so losses must be computed first.
                if self.train_on_pred_boxes:
                    with torch.no_grad():
                        pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                            predictions, proposals
                        )
                        for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                            proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            else:
                losses = {}
            return losses, box_features, predictions[0]
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances
