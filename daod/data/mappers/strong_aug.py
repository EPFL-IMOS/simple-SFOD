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
from daod.data.detection_utils import build_strong_augmentation



class BaseWQDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True
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
        self.strong_augmentation = build_strong_augmentation(None, True)

    def __len__(self):
        return len(self._dataset)

    @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        current_dict = self._dataset.__getitem__(idx)
        img = current_dict["image"]
        device = img.device
        img = img.cpu()
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img = Image.fromarray(img)
        #img = np.array(build_strong_augmentation(img))
        img = np.array(self.strong_augmentation(img))
        cv2.imwrite('/cluster/home/username/adaptive_detectron/RDD-DAOD/save_image/saved.jpg', img)
        img = img.transpose(2, 0, 1)
        img = torch.tensor(img).float()
        img = img.to(device)
        current_dict["image"] = img
        return current_dict
