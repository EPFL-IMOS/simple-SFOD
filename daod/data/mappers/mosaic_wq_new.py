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
from daod.data.detection_utils import build_strong_augmentation
from PIL import Image


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


class MosaicWQNewDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5), shear=2.0, enable_mixup=True,
        mosaic_prob=1.0, mixup_prob=1.0, *args
    ):
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
        self.strong_augmentation = build_strong_augmentation(None, True)

    def __len__(self):
        return len(self._dataset)

    @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            mosaic_labels = []
            #print("Before input dim")
            #print(self._dataset)
            #input_dim = self._dataset.input_dim
            current_dict = self._dataset.__getitem__(idx)
            _, height, width, _, _, _ = current_dict
            height = current_dict[height]
            width = current_dict[width]
            input_dim = (height, width)
            input_h, input_w = input_dim[0], input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            #print(indices)
            mosaic_classes = []
            index_x = np.array([0.5, 0.5, 1.5, 1.5])
            index_y = np.array([0.5, 1.5, 0.5, 1.5])
            index_c = -1
            for i_mosaic, index in enumerate(indices):
                index_c += 1
                # img, _labels, _, img_id = self._dataset.pull_item(index)
                #print('---------------------------')
                #print(self._dataset.__getitem__(index))
                #print('----------------------------')
                #img, _labels, _, img_id = self._dataset.__getitem__(index)
                current_dict = self._dataset.__getitem__(index)
                
                _, height, width, img_id, img, _labels = current_dict
                input_dim = (current_dict[height], current_dict[width])
                input_h, input_w = input_dim[0], input_dim[1]
                #print(str(input_h) + " " + str(input_w))
                # yc, xc = s, s  # mosaic center x, y
                #yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
                #xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))
                yc = int(input_h)
                xc = int(input_w)
                #yc = int(index_y[index_c] * input_h)
                #xc = int(index_x[index_c] * input_w)
                #print("the centers: ")
                #print(yc)
                #print(xc)

                img_id = current_dict[img_id]
                img = current_dict[img]
                #print(img)
                self.device = img.device
                img = img.cpu()
                #print(img)
                img = img.numpy()
                img = img.transpose(1, 2, 0)
                #print(img)
                #print(img.shape)
                _labels = current_dict[_labels]
                origin_labels = _labels
                h0, w0 = img.shape[:2]  # orig hw
                #print(h0)
                #print(w0)
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                img = Image.fromarray(img)
                img = np.array(self.strong_augmentation(img))
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                #print(_labels)
                _labels = _labels.gt_boxes.tensor.cpu().numpy()
                _classes = origin_labels.gt_classes.cpu().numpy()
                for item in _classes:
                    mosaic_classes.append(item)
                #print(_labels)
                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)
                
                #print("current label length: ")
                #print(len(labels))
                #print(labels)
                #print("modified mosaic label length: ")
                #print(len(mosaic_labels))
                #print(mosaic_labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])

            #print(len(mosaic_labels))
            #print("mosaic image shapee")
            #print(mosaic_img.shape)
            '''
            mosaic_img, mosaic_labels = random_affine(
                mosaic_img,
                mosaic_labels,
                target_size=(input_w, input_h),
                degrees=0.0,
                translate=0.0,
                scales=self.scale,
                shear=0.0,
            )
            '''

            height, width = mosaic_img.shape[0], mosaic_img.shape[1]
            new_height, new_width = height // 2, width // 2
            #print(mosaic_img.shape)
            #mosaic_img = mosaic_img.transpose(2, 0, 1)
            #print("before resize: ")
            #print(mosaic_img.shape)
            #print(new_height)
            #print(new_width)
            mosaic_img = cv2.resize(mosaic_img, (new_width, new_height))
            #mosaic_img = Image.fromarray(mosaic_img)
            #mosaic_img = np.array(self.strong_augmentation(mosaic_img))
            #cv2.imwrite('/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/sigao/DA/RDD-DAOD/save_image/' + str(idx) + '.jpg', mosaic_img)
            #print(mosaic_img.shape)
            scale = np.array([0.5, 0.5, 0.5, 0.5])
            mosaic_labels = mosaic_labels * scale

            mosaic_img = mosaic_img.astype(float)
            mosaic_labels = mosaic_labels.astype(float)
            #print(mosaic_labels)
            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            '''
            if (
                self.enable_mixup
                and not len(mosaic_labels) == 0
                and random.random() < self.mixup_prob
            ):
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)
            '''
            #print(mosaic_labels)

            # labels : ()
            #mosaic_labels = np.hstack((mosaic_labels, np.zeros((mosaic_labels.shape[0], 1))))
            #print(mosaic_labels)
            #mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
            #print(padded_labels)
            #img_info = (mix_img.shape[1], mix_img.shape[0])

            # -----------------------------------------------------------------
            # img_info and img_id are not used for training.
            # They are also hard to be specified on a mosaic image.
            # -----------------------------------------------------------------
            # return _, height, width, _, img, _labels 
            dic = dict()
            dic["height"] = height
            dic["width"] = width
            img = mosaic_img.transpose(2, 0, 1)
            #print("img shape: ")
            #print(img.shape)
            img = torch.tensor(img).float()
            img = img.to(self.device)
            dic["image"] = img
            dic["_labels"] = mosaic_labels
            #print(len(mosaic_labels))
            #print(mosaic_labels)
            instances = Instances((height, width))
            instances.gt_boxes = Boxes(torch.tensor(mosaic_labels).float().to(self.device))
            instances.gt_classes = torch.tensor(mosaic_classes).to(self.device)
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

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
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
        origin_labels = box_labels
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        #print("returned labels: ")
        #print(origin_labels)
        return origin_img.astype(np.uint8), origin_labels