#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import random

import cv2
import numpy as np
import torch
from skimage.transform import resize
from detectron2.structures import Boxes, Instances
from yolox.utils import adjust_box_anns, get_local_rank

from yolox.data.data_augment import random_affine
from yolox.data.datasets.datasets_wrapper import Dataset

from PIL import Image, ImageDraw


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class MixupDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5), shear=2.0, enable_mixup=True,
        mosaic_prob=1.0, mixup_prob=1.0, *args
    ):
        print("Init mixup detection")
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            mosaic_scale (tuple):
            mixup_scale (tuple):
            shear (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.local_rank = get_local_rank()

    def __len__(self):
        return len(self._dataset)

    # @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        #print("In mixup get item!")
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            mosaic_labels = []

            current_dict = self._dataset.__getitem__(idx)
            _, height, width, img_id, img, _labels = current_dict
            input_dim = (current_dict[height], current_dict[width])
            input_h, input_w = input_dim[0], input_dim[1]
            yc = int(input_h)
            xc = int(input_w)

            img_id = current_dict[img_id]
            img = current_dict[img]
            self.device = img.device
            img = img.cpu()
            img = img.numpy()
            img = img.transpose(1, 2, 0)
            _labels = current_dict[_labels]
            origin_labels = _labels
            h0, w0 = img.shape[:2]  # orig hw
            scale = min(1. * input_h / h0, 1. * input_w / w0)
            img = cv2.resize(
                img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
            )
            # generate output mosaic image
            (h, w, c) = img.shape[:3]

            #print(_labels)
            _labels = _labels.gt_boxes.tensor.cpu().numpy()
            _labels = scale * _labels
            _classes = origin_labels.gt_classes.cpu().numpy()
            #print(_labels)
            labels = _labels.copy()
            classes = _classes.copy()

            #print(self.enable_mixup)
            #print(len(mosaic_labels))
            #print(self.mixup_prob)
            
            if (
                self.enable_mixup
                and not len(labels) == 0
                and random.random() < self.mixup_prob
            ):
                img, labels, new_classes = self.mixup(img, labels, input_dim)

            #print(classes)
            #print(new_classes)
            #print("afttere mix up")
            for item in new_classes:
                #classes.append(item)
                np.append(classes, item)
            classes = np.append(classes, new_classes)
            #print(labels)
            #print(classes)
            

            dic = dict()
            dic["height"] = height
            dic["width"] = width
            img = img.transpose(2, 0, 1)
            #print("img shape: ")
            #print(img.shape)
            img = torch.tensor(img).float()
            img = img.to(self.device)
            dic["image"] = img
            dic["_labels"] = labels
            #print(len(mosaic_labels))
            #print(mosaic_labels)
            instances = Instances((height, width))
            instances.gt_boxes = Boxes(torch.tensor(labels).float().to(self.device))
            instances.gt_classes = torch.tensor(classes).to(self.device)
            dic["instances"] = instances
            #origin_labels.gt_boxes = Boxes(torch.tensor(mosaic_labels).to(self.device))
            return dic
            #return img, padded_labels, img_info, img_id
            #return img

        else:
            self._dataset._input_dim = self.input_dim
            img, label, img_info, img_id = self._dataset.pull_item(idx)
            img, label = self.preproc(img, label, self.input_dim)
            return img, label, img_info, img_id

    def mixup(self, origin_img, origin_labels, input_dim):
        #print("original bbox: ")
        #print(origin_labels)
        #print("In mixup!")
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            #cp_labels = self._dataset.load_anno(cp_index)
            current_dict = self._dataset.__getitem__(cp_index)
            _, _, _, img_id, img, _labels = current_dict
            cp_labels = current_dict[_labels].gt_boxes.tensor.cpu().numpy()
        #img, cp_labels, _, _ = self._dataset.pull_item(cp_index)
        _, _, _, img_id, img, _labels = self._dataset.__getitem__(cp_index)
        img = current_dict[img].cpu().numpy().transpose(1, 2, 0)
        cp_labels = current_dict[_labels].gt_boxes.tensor.cpu().numpy()
        cp_classes = current_dict[_labels].gt_classes.cpu().numpy()

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        #cp_scale_ratio = 1
        #print("dims")
        #print(input_dim)
        #print(img.shape)
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        #print("resized image shape: ")
        #print(resized_img.shape)

        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        jit_factor = 1
        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        #print("jit image shape: ")
        #print(cp_img.shape)

        #print("cp scale ratio: ")
        #print(cp_scale_ratio)
        #print("jit factor: ")
        #print(jit_factor)
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )

        #cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        #labels = np.hstack((box_labels, cls_labels))
        #origin_labels = np.vstack((origin_labels, labels))
        #origin_labels = box_labels
        #origin_labels = np.append(origin_labels, box_labels)
        mix_labels = []
        for item in origin_labels:
            mix_labels.append(item)
        #print(len(mix_labels))
        
        for item in box_labels:
            mix_labels.append(item)

        '''
        pil_image = Image.fromarray(padded_cropped_img)
        draw = ImageDraw.Draw(pil_image)
        for item in box_labels:
            draw.rectangle(item, outline="red", width=2)
        pil_image.save('/cluster/home/username/adaptive_detectron/RDD-DAOD/save_image/' + str(cp_index) + '_withann.png')
        '''

        
        #print(len(mix_labels))
        mix_labels = np.array(mix_labels)
        '''
        pil_image2 = Image.fromarray(origin_img)
        draw2 = ImageDraw.Draw(pil_image2)
        for item in origin_labels:
            draw2.rectangle(item, outline="red", width=2)
        pil_image2.save('/cluster/home/username/adaptive_detectron/RDD-DAOD/save_image/' + str(cp_index) + '_ori.png')
        '''
        origin_img = origin_img.astype(np.float32)

        #cv2.imwrite('/cluster/home/username/adaptive_detectron/RDD-DAOD/save_image/' + str(cp_index) + '_origin.jpg', origin_img)
        #cv2.imwrite('/cluster/home/username/adaptive_detectron/RDD-DAOD/save_image/' + str(cp_index) + '_padded.jpg', padded_cropped_img)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)
        #cv2.imwrite('/cluster/home/username/adaptive_detectron/RDD-DAOD/save_image/' + str(cp_index) + '_result.jpg', origin_img)
        origin_img = origin_img.astype(np.uint8)
        
        pil_image2 = Image.fromarray(origin_img)
        draw2 = ImageDraw.Draw(pil_image2)
        for item in mix_labels:
            draw2.rectangle(item, outline="red", width=2)
        #pil_image2.save('/cluster/home/username/adaptive_detectron/RDD-DAOD/save_image/' + str(cp_index) + '_ori.png')
       
        return origin_img.astype(np.uint8), mix_labels, cp_classes