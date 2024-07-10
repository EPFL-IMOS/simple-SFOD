# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from typing import Dict, Tuple, List, Optional, Union

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
from detectron2.structures import ImageList

from daod.modeling.dann import gradient_scalar, FCDiscriminator_img, DAInsHead
from daod.loss.bpc_loss import bpc_loss


@META_ARCH_REGISTRY.register()
class SourceFreeAdaptiveTeacherGeneralizedRCNN(GeneralizedRCNN):

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
        dis_type: str,
        ins_dc: bool
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

        self.dis_type = dis_type
        self.DC_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]) # Need to know the channel
        self.ins_dc = ins_dc
        if self.ins_dc:
            self.DC_ins = DAInsHead(self.roi_heads.box_predictor.cls_score.in_features, [self.dis_type])
            #print(self.DC_ins)
        

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        cls.cfg = cfg
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "dis_type": cfg.SEMISUPNET.DIS_TYPE,
            "ins_dc": cfg.SEMISUPNET.INS_DC,
            # "dis_loss_ratio": cfg.xxx,
        }

    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t

    def forward(self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False):
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
        if (not self.training) and (not val_mode):  # only conduct when testing mode
            return self.inference(batched_inputs)

        # print("be in forward function")

        source_label = 0
        target_label = 1

        if branch == "domain_classifier":
            images_s, images_t = self.preprocess_image_train(batched_inputs)

            if "instances" in batched_inputs[0]:
                gt_instances_s = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances_s = None

            features_s = self.backbone(images_s.tensor)

            features_s_reversed = gradient_scalar(features_s[self.dis_type], -1.0)
            DC_img_out_s = self.DC_img(features_s_reversed)
            loss_DC_img_s = F.binary_cross_entropy_with_logits(DC_img_out_s, torch.FloatTensor(DC_img_out_s.data.size()).fill_(source_label).to(self.device))

            features_t = self.backbone(images_t.tensor)
            
            features_t_reversed = gradient_scalar(features_t[self.dis_type], -1.0)
            DC_img_out_t = self.DC_img(features_t_reversed)
            loss_DC_img_t = F.binary_cross_entropy_with_logits(DC_img_out_t, torch.FloatTensor(DC_img_out_t.data.size()).fill_(target_label).to(self.device))

            if self.ins_dc:

                proposals_rpn_s, _ = self.proposal_generator(
                    images_s, features_s, None, compute_loss=False
                )
                proposals_s, _, box_features_s, _= self.roi_heads(
                    images_s,
                    features_s,
                    proposals_rpn_s,
                    targets=gt_instances_s,
                    compute_loss=True,
                    branch=branch,
                )

                if "instances_unlabeled" in batched_inputs[0]:
                    gt_instances_t = [x["instances_unlabeled"].to(self.device) for x in batched_inputs]
                else:
                    gt_instances_t = None

                proposals_rpn_t, _ = self.proposal_generator(
                    images_t, features_t, None, compute_loss=False
                )
                proposals_t, _, box_features_t, _= self.roi_heads(
                    images_t,
                    features_t,
                    proposals_rpn_t,
                    targets=gt_instances_t,
                    compute_loss=True,
                    branch=branch,
                )
                #print("proposals s")
                #print(proposals_s)
                boxes = [x.proposal_boxes for x in proposals_s]
                level_assignments_s = assign_boxes_to_levels(boxes, self.roi_heads.box_pooler.min_level,
                    self.roi_heads.box_pooler.max_level, self.roi_heads.box_pooler.canonical_box_size, self.roi_heads.box_pooler.canonical_level)
                boxes = [x.proposal_boxes for x in proposals_t]
                level_assignments_t = assign_boxes_to_levels(boxes, self.roi_heads.box_pooler.min_level,
                    self.roi_heads.box_pooler.max_level, self.roi_heads.box_pooler.canonical_box_size, self.roi_heads.box_pooler.canonical_level)

                # Source instance-level discriminator
                loss_DC_ins_s = self.instance_dc_loss(box_features_s, level_assignments_s, source_label)
                #print(loss_DC_ins_s)

                # Target instance-level discriminator
                loss_DC_ins_t = self.instance_dc_loss(box_features_t, level_assignments_t, target_label)
                #print(loss_DC_ins_t)

            losses = {}
            losses["loss_DC_img_s"] = loss_DC_img_s
            losses["loss_DC_img_t"] = loss_DC_img_t
            if self.ins_dc:
                losses["loss_DC_ins_s"] = loss_DC_ins_s
                losses["loss_DC_ins_t"] = loss_DC_ins_t
            return losses, [], []

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            #gt_instances = []
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            #for x in batched_inputs:
                #instances = x["instances"].to(self.device)
                #for item in instances:
                #    item["image_id"] = x["image_id"]
                #gt_instances.append(instances)
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            features_s = gradient_scalar(features[self.dis_type], -1.0)
            DC_img_out_s = self.DC_img(features_s)
            loss_DC_img_s = F.binary_cross_entropy_with_logits(DC_img_out_s, torch.FloatTensor(DC_img_out_s.data.size()).fill_(source_label).to(self.device))

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses, _, _= self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses["loss_DC_img_s"] = loss_DC_img_s*0.001
            return losses, [], []

        elif branch == "supervised_target": 

            # Region proposal network

            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses, _, proposal_instances= self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # print(box_predictions)

            # print(proposals_rpn)
            proposals_roih, _ = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            # print(proposals_roih)
            
            # self_tcd_loss = tcd_loss(proposals_roih, gt_instances)
            # tcd_loss = TcdLoss(self.cfg.MODEL.ROI_HEADS.NUM_CLASSES)
            bpc_losses = bpc_loss(self.cfg.MODEL.ROI_HEADS.NUM_CLASSES, gt_instances, proposal_instances)
            #print(bpc_losses)
            

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # losses["loss_DC_img_t"] = loss_DC_img_t*0.001
            # losses["loss_DC_img_s"] = loss_DC_img_s*0.001
            
            #print(losses)
            #print(proposals_roih)
            losses["loss_bpc"] = bpc_losses
            return losses, proposals_roih, [], []

        elif branch == "unsup_data_weak":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # Region proposal network
            #print("before")
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )
            #print("after")

            # print(proposals_rpn)
            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            #for p in proposals_rpn:
            #    print(p.gt_classes)
            proposals_roih, _ = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch
            )

            return {}, proposals_rpn, proposals_roih

    def instance_dc_loss(self, box_features, level_assignments, domain_label):
        """Instance-level domain classifier loss over multiple feature levels."""
        box_features_reversed = gradient_scalar(box_features, -1.0)
        #print(box_features_reversed)
        #print(box_features_reversed.size())
        DC_ins_out = self.DC_ins(box_features_reversed, levels=level_assignments)
        #print("DC_ins_out:", DC_ins_out.size())
        domain_label_t = torch.FloatTensor(DC_ins_out.data.size()).fill_(domain_label).to(self.device)
        return F.binary_cross_entropy_with_logits(DC_ins_out, domain_label_t)

    def visualize_training(self, batched_inputs, proposals, branch=""):
        """
        This function different from the original one:
        - it adds "branch" to the `vis_name`.

        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes.tensor.detach().cpu().numpy())
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = (
                "Left: GT bounding boxes "
                + branch
                + ";  Right: Predicted proposals "
                + branch
            )
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

