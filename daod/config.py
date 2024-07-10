# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


AVAILABLE_TRAINERS = ["da", "adaptive_teacher", "source_free_adaptive_teacher"]


def add_config(cfg):
    """
    Add configuration keys for all trainers.
    """
    _C = cfg
    _C.TRAINER = ""
    _C.TEST.IMS_PER_BATCH = 1
    _C.DATASETS.TRAIN_TARGET = ()
    _C.SOLVER.IMS_PER_BATCH_TARGET = 1
    _C.TEST.VAL_LOSS = True
    
    # if the vgg model hass batchnorm layers
    _C.VGG = CN()
    _C.VGG.BN = True


    for trainer in AVAILABLE_TRAINERS:
        add_trainer_config(cfg, trainer)


def add_trainer_config(cfg, trainer):
    """
    Add configuration for specific trainers.
    """
    _C = cfg

    if trainer == "da":
        _C.DA_FASTER = CN()
        _C.DA_FASTER.DC_IMG_GRL_WEIGHT = 0.01
        _C.DA_FASTER.DC_INS_GRL_WEIGHT = 0.1
        _C.DA_FASTER.DC_CONSISTENCY_WEIGHT = 0.1
        _C.DA_FASTER.LEVELS = ["res4"]
        _C.DA_FASTER.ENTROPY_CONDITIONING = False

    elif trainer == "adaptive_teacher":
        _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
        _C.MODEL.RPN.LOSS = "CrossEntropy"
        _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"

        _C.SOLVER.FACTOR_LIST = (1,)

        _C.TEST.EVALUATOR = "COCOeval"

        _C.SEMISUPNET = CN()

        # Output dimension of the MLP projector after `res5` block
        _C.SEMISUPNET.MLP_DIM = 128

        # Semi-supervised training
        # _C.SEMISUPNET.Trainer = "ateacher"
        _C.SEMISUPNET.BBOX_THRESHOLD = 0.7
        _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
        _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
        _C.SEMISUPNET.BURN_UP_STEP = 12000
        _C.SEMISUPNET.EMA_KEEP_RATE = 0.0
        _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
        _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
        _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"
        _C.SEMISUPNET.DIS_TYPE = "res4"
        _C.SEMISUPNET.DIS_LOSS_WEIGHT = 0.1
        _C.SEMISUPNET.INS_DC = False

        # dataloader
        # supervision level
        _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
        _C.DATALOADER.RANDOM_DATA_SEED = 0  # random seed to read data
        # _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/COCO_supervision.txt"

        _C.EMAMODEL = CN()
        _C.EMAMODEL.SUP_CONSIST = True


    elif trainer == "source_free_adaptive_teacher":
        _C.MODEL.RPN.UNSUP_LOSS_WEIGHT = 1.0
        _C.MODEL.RPN.LOSS = "CrossEntropy"
        _C.MODEL.ROI_HEADS.LOSS = "CrossEntropy"

        _C.SOLVER.FACTOR_LIST = (1,)

        _C.TEST.EVALUATOR = "COCOeval"

        _C.SEMISUPNET = CN()

        # Output dimension of the MLP projector after `res5` block
        _C.SEMISUPNET.MLP_DIM = 128

        # Semi-supervised training
        # _C.SEMISUPNET.Trainer = "ateacher"
        _C.SEMISUPNET.BBOX_THRESHOLD = 0.7
        _C.SEMISUPNET.PSEUDO_BBOX_SAMPLE = "thresholding"
        _C.SEMISUPNET.TEACHER_UPDATE_ITER = 1
        _C.SEMISUPNET.BURN_UP_STEP = 12000
        _C.SEMISUPNET.EMA_KEEP_RATE = 0.0
        _C.SEMISUPNET.UNSUP_LOSS_WEIGHT = 4.0
        _C.SEMISUPNET.SUP_LOSS_WEIGHT = 0.5
        _C.SEMISUPNET.LOSS_WEIGHT_TYPE = "standard"
        _C.SEMISUPNET.DIS_TYPE = "res4"
        _C.SEMISUPNET.DIS_LOSS_WEIGHT = 0.1
        _C.SEMISUPNET.INS_DC = False

        # dataloader
        # supervision level
        _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
        _C.DATALOADER.RANDOM_DATA_SEED = 0  # random seed to read data
        # _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/COCO_supervision.txt"

        _C.EMAMODEL = CN()
        _C.EMAMODEL.SUP_CONSIST = True

        # print("until now it's all good")

        _C.ADAPTIVE_THRESHOLD = CN()
        # if use adaptive threshold or not
        _C.ADAPTIVE_THRESHOLD.ENABLED = True

        # adaptive threshold warm up ites
        _C.ADAPTIVE_THRESHOLD.WARM_UP = 100

        _C.ADAPTIVE_THRESHOLD.RESERVE = 500

        # choose from if the data fed into the student network is strong augmentation or style enhancement module
        _C.WEAK_STRONG_AUGMENT = True
        _C.ENHANCE = True

        # choose if the domain classifier is used or not
        _C.DOMAIN_CLASSIFIER = CN()
        _C.DOMAIN_CLASSIFIER.ENABLED = False
        _C.DOMAIN_CLASSIFIER.IMAGE = False
        _C.DOMAIN_CLASSIFIER.INSTANCE = False

        _C.STYLE = CN()
        _C.STYLE.ENABLED = False
        _C.STYLE.STYLE_IMAGE = None
        _C.STYLE.VGG_MODEL = None
        _C.STYLE.DECODER = None