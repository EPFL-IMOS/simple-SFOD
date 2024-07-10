# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, Tuple, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.poolers import assign_boxes_to_levels
from detectron2.utils.events import get_event_storage

from daod.modeling.dann import gradient_scalar, DAImgHead, DAInsHead
from daod.modeling.utils import entropy


class MultiLinearMap(nn.Module):
    """Multi linear map
    Shape:
        - f: (minibatch, F)
        - g: (minibatch, C)
        - Outputs: (minibatch, F * C)
    """

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch_size = f.size(0)
        output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        return output.view(batch_size, -1)


@META_ARCH_REGISTRY.register()
class CDAFasterRCNN(GeneralizedRCNN):
    """Conditional DA-Faster-RCNN."""
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        levels: List[str],
        pooler_resolution: int,
        dc_img_grl_weight: float,
        dc_ins_grl_weight: float,
        dc_consistency_weight: float,
        entropy_conditioning: bool
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
            levels: list of image feature levels.
        """
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        self.levels = levels
        self.pooler_resolution = pooler_resolution
        self.dc_img_grl_weight = dc_img_grl_weight
        self.dc_ins_grl_weight = dc_ins_grl_weight
        self.dc_consistency_weight = dc_consistency_weight
        self.entropy_conditioning = entropy_conditioning
        self.DC_img = DAImgHead(self.backbone._out_feature_channels[self.levels[0]], self.levels).to(self.device)
        self.DC_ins = DAInsHead(self.roi_heads.box_predictor.cls_score.in_features * self.roi_heads.box_predictor.cls_score.out_features, self.levels).to(self.device)
        self.multilinear_map = MultiLinearMap()
    
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "levels": cfg.DA_FASTER.LEVELS,
            "pooler_resolution": cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
            "dc_img_grl_weight": cfg.DA_FASTER.DC_IMG_GRL_WEIGHT,
            "dc_ins_grl_weight": cfg.DA_FASTER.DC_INS_GRL_WEIGHT,
            "dc_consistency_weight": cfg.DA_FASTER.DC_CONSISTENCY_WEIGHT,
            "entropy_conditioning": cfg.DA_FASTER.ENTROPY_CONDITIONING
        }

    def forward(self, batched_inputs: Union[List[Dict], Tuple[List[Dict]]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        if isinstance(batched_inputs, tuple):
            batched_inputs_s, batched_inputs_t = batched_inputs
        else:
            batched_inputs_s, batched_inputs_t = batched_inputs, None

        ### Supervised training ###
        losses = {}

        images_s = self.preprocess_image(batched_inputs_s)

        if "instances" in batched_inputs_s[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs_s]
        else:
            gt_instances = None

        # Source image-level features
        img_features_s = self.backbone(images_s.tensor)

        # Source region proposals
        proposals_rpn_s, proposal_losses = self.proposal_generator(
            images_s, img_features_s, gt_instances
        )

        # Source roi_head lower branch
        proposals_s, detector_losses, box_features_s, box_scores_s = self.roi_heads(
            images_s,
            img_features_s,
            proposals_rpn_s,
            targets=gt_instances,
        )

        losses.update(detector_losses)
        losses.update(proposal_losses)

        ### Domain-adversarial training ###
        if batched_inputs_t is not None:

            source_label = 0
            target_label = 1

            images_t = self.preprocess_image(batched_inputs_t)

            # Target image-level features
            img_features_t = self.backbone(images_t.tensor)

            # Source image-level discriminator
            loss_dc_img_s = self.image_dc_loss(img_features_s, source_label)

            # Target image-level discriminator
            loss_dc_img_t = self.image_dc_loss(img_features_t, target_label)

            # Target region proposals
            proposals_rpn_t, _ = self.proposal_generator(
                images_t, img_features_t
            )

            # Target roi_head lower branch
            proposals_t, _, box_features_t, box_scores_t = self.roi_heads(
                images_t,
                img_features_t,
                proposals_rpn_t
            )

            # print("IMG FEATURES")
            # print(img_features_s[self.dis_type].size())
            # print(img_features_t[self.dis_type].size())
            # print("PROPOSALS")
            # print(len(proposals_rpn_s[0]))
            # print(len(proposals_rpn_t[0]))
            # print(proposals_rpn_s[0])
            # print(proposals_rpn_t[0])
            # print("BOX FEATURES")
            # print(box_features_s.size())
            # print(box_features_t.size())
            # print(box_features_s)
            # print(box_features_t)

            boxes = [x.proposal_boxes for x in proposals_s]
            level_assignments_s = assign_boxes_to_levels(boxes, self.roi_heads.box_pooler.min_level,
                self.roi_heads.box_pooler.max_level, self.roi_heads.box_pooler.canonical_box_size, self.roi_heads.box_pooler.canonical_level)
            boxes = [x.proposal_boxes for x in proposals_t]
            level_assignments_t = assign_boxes_to_levels(boxes, self.roi_heads.box_pooler.min_level,
                self.roi_heads.box_pooler.max_level, self.roi_heads.box_pooler.canonical_box_size, self.roi_heads.box_pooler.canonical_level)

            # Source conditional instance-level discriminator loss
            loss_dc_ins_s = self.conditional_instance_dc_loss(box_features_s, box_scores_s, level_assignments_s, source_label)

            # Target conditional instance-level discriminator loss
            loss_dc_ins_t = self.conditional_instance_dc_loss(box_features_t, box_scores_t, level_assignments_t, target_label)

            # Source consistency loss
            loss_dc_consistency_s = self.conditional_consistency_loss(img_features_s, box_features_s, box_scores_s, proposals_s, level_assignments_s)

            # Target consistency loss
            loss_dc_consistency_t = self.conditional_consistency_loss(img_features_t, box_features_t, box_scores_t, proposals_t, level_assignments_t)

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs_s, proposals_rpn_s, domain="source")
                    self.visualize_training(batched_inputs_t, proposals_rpn_t, domain="target")

            losses["loss_DC_img"] = 0.5*(loss_dc_img_s + loss_dc_img_t)
            losses["loss_DC_ins"] = 0.5*(loss_dc_ins_s + loss_dc_ins_t)
            losses["loss_DC_consistency"] = 0.5*(loss_dc_consistency_s + loss_dc_consistency_t)

        return losses

    def image_dc_loss(self, img_features, domain_label):
        """Image-level domain classifier loss over multiple feature levels."""
        img_features_reversed = {level: gradient_scalar(img_features[level], -1.0*self.dc_img_grl_weight) for level in self.levels}
        dc_img_out = self.DC_img(img_features_reversed)#.flatten(start_dim=1)
        upsampled_losses = []
        _, _, H, W = dc_img_out[self.levels[0]].size()
        # print(f"H, W = {H}, {W}")
        for level in self.levels:
            out = dc_img_out[level]
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=True)
            domain_label_t = torch.FloatTensor(out.data.size()).fill_(domain_label).to(self.device)
            level_loss = F.binary_cross_entropy_with_logits(out, domain_label_t, reduction="none")
            upsampled_losses.append(level_loss)
        return torch.stack(upsampled_losses).mean()

    def conditional_instance_dc_loss(self, box_features, box_scores, level_assignments, domain_label):
        """Conditional instance-level domain classifier loss over multiple feature levels."""
        # print("box_features:", box_features.size())
        # print(box_features)
        # print("box_scores:", box_scores.size())
        # print(box_scores)
        condition = F.softmax(box_scores, dim=1).detach()
        conditioned_box_features_reversed = gradient_scalar(self.multilinear_map(box_features, condition), -1.0*self.dc_ins_grl_weight)
        # print("conditioned_box_features_reversed:", conditioned_box_features_reversed.size())
        # print(conditioned_box_features_reversed)
        dc_ins_out = self.DC_ins(conditioned_box_features_reversed, levels=level_assignments)
        # print("dc_ins_out:", dc_ins_out.size())
        domain_label_t = torch.FloatTensor(dc_ins_out.data.size()).fill_(domain_label).to(self.device)
        if self.entropy_conditioning:
            weight = 1.0 + torch.exp(-entropy(condition))
            # print("weight:", weight)
            weight = weight / weight.mean()
            # print("weight:", weight)
            return F.binary_cross_entropy_with_logits(dc_ins_out, domain_label_t, weight.view_as(dc_ins_out))
        else:
            return F.binary_cross_entropy_with_logits(dc_ins_out, domain_label_t)

    def conditional_consistency_loss(self, img_features, box_features, box_scores, proposals, level_assignments):
        """Consistency loss between image-level and instance-level domain classifier predictions."""
        img_features = {level: gradient_scalar(img_features[level], self.dc_consistency_weight*self.dc_img_grl_weight) for level in self.levels}
        dc_img_out = self.DC_img(img_features)
        dc_img_probs = {level: dc_img_out[level].sigmoid() for level in self.levels}
        dc_img_rois_probs = self.roi_heads.box_pooler([dc_img_probs[level] for level in self.levels], [x.proposal_boxes for x in proposals])
        dc_img_rois_probs = F.avg_pool2d(dc_img_rois_probs, kernel_size=self.pooler_resolution, stride=self.pooler_resolution)

        condition = F.softmax(box_scores, dim=1).detach()
        conditioned_box_features = gradient_scalar(self.multilinear_map(box_features, condition), self.dc_consistency_weight*self.dc_ins_grl_weight)
        dc_ins_out = self.DC_ins(conditioned_box_features, levels=level_assignments)
        dc_ins_probs = dc_ins_out.sigmoid()

        consistency_loss = F.l1_loss(dc_img_rois_probs.squeeze(), dc_ins_probs.squeeze())

        return consistency_loss
    
    def visualize_training(self, batched_inputs, proposals, domain=""):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
            domain (str): name of the domain for display.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = f"[{domain}] Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
