# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import operator
import json
import torch.utils.data
from detectron2.utils.comm import get_world_size
from detectron2.data.common import (
    DatasetFromList,
    MapDataset,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import (
    InferenceSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from detectron2.data.build import (
    trivial_batch_collator,
    worker_init_reset_seed,
    get_detection_dataset_dicts,
    build_batch_data_loader,
)
from daod.data.common import AspectRatioGroupedSemiSupDatasetTwoCrop, AspectRatioGroupedSemiSupDataset, AspectRatioGroupedSemiSupDatasetTwoCropSourceFree


"""
This file contains the default logic to build a dataloader for training or testing.
"""

def divide_label_unlabel(
    dataset_dicts, SupPercent, random_data_seed, random_data_seed_path
):
    num_all = len(dataset_dicts)
    num_label = int(SupPercent / 100.0 * num_all)

    # read from pre-generated data seed
    with open(random_data_seed_path) as COCO_sup_file:
        coco_random_idx = json.load(COCO_sup_file)

    labeled_idx = np.array(coco_random_idx[str(SupPercent)][str(random_data_seed)])
    assert labeled_idx.shape[0] == num_label, "Number of READ_DATA is mismatched."

    label_dicts = []
    unlabel_dicts = []
    labeled_idx = set(labeled_idx)

    for i in range(len(dataset_dicts)):
        if i in labeled_idx:
            label_dicts.append(dataset_dicts[i])
        else:
            unlabel_dicts.append(dataset_dicts[i])

    return label_dicts, unlabel_dicts


# uesed by supervised-only baseline trainer
def build_detection_semisup_train_loader(cfg, mapper=None):

    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    # Divide into labeled and unlabeled sets according to supervision percentage
    label_dicts, unlabel_dicts = divide_label_unlabel(
        dataset_dicts,
        cfg.DATALOADER.SUP_PERCENT,
        cfg.DATALOADER.RANDOM_DATA_SEED,
        cfg.DATALOADER.RANDOM_DATA_SEED_PATH,
    )

    dataset = DatasetFromList(label_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))

    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = (
            RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                label_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
            )
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    # list num of labeled and unlabeled
    logger.info("Number of training samples " + str(len(dataset)))
    logger.info("Supervision percentage " + str(cfg.DATALOADER.SUP_PERCENT))

    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


# uesed by evaluation
def build_detection_test_loader(cfg, dataset_name, mapper=None):
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[
                list(cfg.DATASETS.TEST).index(dataset_name)
            ]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


# used by unbiased teacher trainer
def build_detection_semisup_train_loader_two_crops(cfg, mapper=None):
    if "TRAIN_TARGET" in cfg.DATASETS:  # cross-dataset (e.g., coco-additional)
        label_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
        unlabel_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_TARGET,
            filter_empty=False,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )           
    else:  # different degree of supervision (e.g., COCO-supervision)
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )

        # Divide into labeled and unlabeled sets according to supervision percentage
        label_dicts, unlabel_dicts = divide_label_unlabel(
            dataset_dicts,
            cfg.DATALOADER.SUP_PERCENT,
            cfg.DATALOADER.RANDOM_DATA_SEED,
            cfg.DATALOADER.RANDOM_DATA_SEED_PATH,
        )

    label_dataset = DatasetFromList(label_dicts, copy=False)
    # exclude the labeled set from unlabeled dataset
    unlabel_dataset = DatasetFromList(unlabel_dicts, copy=False)
    # include the labeled set in unlabel dataset
    # unlabel_dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    label_dataset = MapDataset(label_dataset, mapper)
    unlabel_dataset = MapDataset(unlabel_dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        label_sampler = TrainingSampler(len(label_dataset))
        unlabel_sampler = TrainingSampler(len(unlabel_dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        raise NotImplementedError("{} not yet supported.".format(sampler_name))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return build_semisup_batch_data_loader_two_crop(
        (label_dataset, unlabel_dataset),
        (label_sampler, unlabel_sampler),
        cfg.SOLVER.IMS_PER_BATCH,
        cfg.SOLVER.IMS_PER_BATCH_TARGET,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


# batch data loader
def build_semisup_batch_data_loader_two_crop(
    dataset,
    sampler,
    total_batch_size_label,
    total_batch_size_unlabel,
    *,
    aspect_ratio_grouping=False,
    num_workers=0
):
    world_size = get_world_size()
    assert (
        total_batch_size_label > 0 and total_batch_size_label % world_size == 0
    ), "Total label batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    assert (
        total_batch_size_unlabel > 0 and total_batch_size_unlabel % world_size == 0
    ), "Total unlabel batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    batch_size_label = total_batch_size_label // world_size
    batch_size_unlabel = total_batch_size_unlabel // world_size

    label_dataset, unlabel_dataset = dataset
    label_sampler, unlabel_sampler = sampler

    if aspect_ratio_grouping:
        label_data_loader = torch.utils.data.DataLoader(
            label_dataset,
            sampler=label_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        unlabel_data_loader = torch.utils.data.DataLoader(
            unlabel_dataset,
            sampler=unlabel_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return AspectRatioGroupedSemiSupDatasetTwoCrop(
            (label_data_loader, unlabel_data_loader),
            (batch_size_label, batch_size_unlabel),
        )
    else:
        raise NotImplementedError("ASPECT_RATIO_GROUPING = False is not supported yet")


# used by source free unbiased teacher trainer
def build_detection_semisup_train_loader_two_crops_source_free(cfg, mapper=None):
    if "TRAIN_TARGET" in cfg.DATASETS:  # cross-dataset (e.g., coco-additional)
        unlabel_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_TARGET,
            filter_empty=False,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )

    unlabel_dataset = DatasetFromList(unlabel_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    unlabel_dataset = MapDataset(unlabel_dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        unlabel_sampler = TrainingSampler(len(unlabel_dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        raise NotImplementedError("{} not yet supported.".format(sampler_name))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return build_semisup_batch_data_loader_two_crop_source_free(
        unlabel_dataset,
        unlabel_sampler,
        cfg.SOLVER.IMS_PER_BATCH_TARGET,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


# batch data loader
def build_semisup_batch_data_loader_two_crop_source_free(
    dataset,
    sampler,
    total_batch_size_unlabel,
    *,
    aspect_ratio_grouping=False,
    num_workers=0
):
    world_size = get_world_size()

    assert (
        total_batch_size_unlabel > 0 and total_batch_size_unlabel % world_size == 0
    ), "Total unlabel batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_unlabel, world_size
    )

    batch_size_unlabel = total_batch_size_unlabel // world_size

    unlabel_dataset = dataset
    unlabel_sampler = sampler

    if aspect_ratio_grouping:
        #print(unlabel_dataset[0][0])
        #print(unlabel_dataset[0][0]['image'].device)
        unlabel_data_loader = torch.utils.data.DataLoader(
            unlabel_dataset,
            sampler=unlabel_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        #print(next(unlabel_data_loader)[0]['image'].device)
        return AspectRatioGroupedSemiSupDatasetTwoCropSourceFree(
            unlabel_data_loader,
            batch_size_unlabel
        )
    else:
        raise NotImplementedError("ASPECT_RATIO_GROUPING = False is not supported yet")

# used by da faster trainer
def build_detection_da_train_loader(cfg, mapper=None):
    source_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    target_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN_TARGET,
        filter_empty=False,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN_TARGET
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    source_dataset = DatasetFromList(source_dicts, copy=False)
    target_dataset = DatasetFromList(target_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    source_dataset = MapDataset(source_dataset, mapper)
    target_dataset = MapDataset(target_dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        source_sampler = TrainingSampler(len(source_dataset))
        target_sampler = TrainingSampler(len(target_dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        raise NotImplementedError("{} not yet supported.".format(sampler_name))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return build_da_batch_data_loader(
        (source_dataset, target_dataset),
        (source_sampler, target_sampler),
        cfg.SOLVER.IMS_PER_BATCH,
        cfg.SOLVER.IMS_PER_BATCH_TARGET,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


# batch data loader
def build_da_batch_data_loader(
    dataset,
    sampler,
    total_batch_size_source,
    total_batch_size_target,
    *,
    aspect_ratio_grouping=False,
    num_workers=0
):
    world_size = get_world_size()
    assert (
        total_batch_size_source > 0 and total_batch_size_source % world_size == 0
    ), "Total source batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_source, world_size
    )

    assert (
        total_batch_size_target > 0 and total_batch_size_target % world_size == 0
    ), "Total target batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_source, world_size
    )

    batch_size_source = total_batch_size_source // world_size
    batch_size_target = total_batch_size_target // world_size

    source_dataset, target_dataset = dataset
    source_sampler, target_sampler = sampler

    if aspect_ratio_grouping:
        source_data_loader = torch.utils.data.DataLoader(
            source_dataset,
            sampler=source_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        target_data_loader = torch.utils.data.DataLoader(
            target_dataset,
            sampler=target_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return AspectRatioGroupedSemiSupDataset(
            (source_data_loader, target_data_loader),
            (batch_size_source, batch_size_target),
        )
    else:
        raise NotImplementedError("ASPECT_RATIO_GROUPING = False is not supported yet")