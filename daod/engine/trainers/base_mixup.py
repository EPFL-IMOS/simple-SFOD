"""
Base trainer class.
"""
import os
import time
import numpy as np
import operator

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import torch.utils.data as torchdata

import detectron2.utils.comm as comm
from detectron2.utils.env import seed_all_rng
from detectron2.config import configurable
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase, hooks
from detectron2.engine.train_loop import AMPTrainer
from detectron2.evaluation import DatasetEvaluators, COCOEvaluator, PascalVOCDetectionEvaluator
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.common import ToIterableDataset, AspectRatioGroupedDataset, DatasetFromList, MapDataset
from detectron2.data.build import _train_loader_from_config
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.events import EventStorage
from detectron2.data.samplers import TrainingSampler
from detectron2.utils.comm import get_world_size
import logging

from daod.engine.hooks import ValLossHook
from daod.evaluation import F1Evaluator, NewCOCOEvaluator, DECE, SimCOCOEvaluator
from daod.data import DatasetMapperAnnotation
from daod.data.mappers import MixupDetection

from yolox.data import TrainTransform


class BaseMixupTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """Base trainer class.
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        hooks = self.build_hooks()
        self.register_hooks(hooks)

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[BaseTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        record_dict = self.model(data)

        # num_gt_bbox = 0.0
        # for element in data:
        #     num_gt_bbox += len(element["instances"])
        # num_gt_bbox = num_gt_bbox / len(data)
        # record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        #print(dataset_name)
        if evaluator_type == "coco":
            #print("evaluator type coco")
            #print(dataset_name[0:6])
            if dataset_name[0:6] == "sim10k" or dataset_name[0:5] == "kitti":
                #print("in if")
                evaluator_list.append(SimCOCOEvaluator(
                    dataset_name, output_dir=output_folder))
            else:
                evaluator_list.append(NewCOCOEvaluator(
                    dataset_name, output_dir=output_folder))
            #evaluator_list.append(NewCOCOEvaluator(
            #    dataset_name, output_dir=output_folder))         
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_6classes":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        evaluator_list.append(F1Evaluator(cfg.MODEL.ROI_HEADS.NUM_CLASSES))
        #evaluator_list.append(DECE(cfg.MODEL.ROI_HEADS.NUM_CLASSES))

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "No Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        Override of :func:`detectron2.data.build_detection_test_loader` with custom batch size instead of 1.
        """
        return build_detection_test_loader(cfg, dataset_name, batch_size=cfg.TEST.IMS_PER_BATCH, mapper=DatasetMapperAnnotation(cfg, is_train=False))
        # return build_detection_test_loader(cfg, dataset_name, batch_size=cfg.TEST.IMS_PER_BATCH, mapper=DatasetMapperAnnotationMixUp(cfg, is_train=False))

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg)
        # return build_detection_train_loader(cfg, mapper=DatasetMapperAnnotationMixUp(cfg, is_train=True))

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        if cfg.TEST.VAL_LOSS:
            ret.append(ValLossHook(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=DatasetMapper(self.cfg, is_train=True))
            ))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

def test_refinement(cfg, model, evaluators=None):
    trainer = BaseTrainer(cfg)
    model.train()
    trainer.training = True
    print(trainer.training)
    i = 0
    print("---------------------------------")
    print("In test refinement")
    print("---------------------------------")
    with EventStorage(trainer.start_iter) as trainer.storage:
        while(True):
            i += 1
            data = next(trainer._trainer._data_loader_iter)
            #print(data)
            if (data):
                # print(data)
                # print("here")
                with torch.no_grad():
                    result = model(data)
            else:
                break

            if i > 1000:
                break
    test_results = trainer.test(cfg, model)
    checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)

    checkpointer.save("adabn")

    '''
    new_model = trainer.build_model(cfg)
    DetectionCheckpointer(new_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        "/cluster/scratch/username/weights/adabn_weight/adabn.pth", resume=False
    )

    new_test_results = trainer.test(cfg, new_model)
    print("new results")
    print(new_test_results)
    '''
    return test_results

@classmethod
def reset_bn_stats(self, module):
    """Reset running statistics in the BatchNorm layers."""
    if isinstance(module, nn.BatchNorm2d):
        print(module)
        module.running_mean = nn.Parameter(torch.zeros_like(module.running_mean), requires_grad=False)
        module.running_var = nn.Parameter(torch.ones_like(module.running_var), requires_grad=False)

def recursive_traversal(trainer, module):
    for child in module.children():
        trainer.reset_bn_stats(child)
        trainer.recursive_traversal(child)

def adabn_refinement(cfg, model, evaluators=None):
    recursive_traversal(model)
    recursive_traversal(model)
    # replace all the batch statistics

    # we will use the target - training data
    # calculate the statistics for the full datasets
    return test_refinement(cfg, model)

@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset,
    *,
    mapper,
    sampler=None,
    total_batch_size,
    aspect_ratio_grouping=True,
    num_workers=0,
    collate_fn=None,
    prefetch_factor=None,
    persistent_workers=False,
    pin_memory=False,
):
    """
    Build a dataloader for object detection with some default features.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). It can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``.
            If ``dataset`` is map-style, the default sampler is a :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
            Sampler must be None if ``dataset`` is iterable.
        total_batch_size (int): total batch size across all workers.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers
        collate_fn: a function that determines how to do batching, same as the argument of
            `torch.utils.data.DataLoader`. Defaults to do no collation and return a list of
            data. No collation is OK for small batch size and simple data structures.
            If your batch size is large and each sample contains too many small tensors,
            it's more efficient to collate them in data loader.

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """
    # mapper = None
    '''
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    '''

    #print(dataset)

    '''
    input_size = (640, 640)
    mosaic_prob = 1.0
    # prob of applying mixup aug
    mixup_prob = 1.0
    # prob of applying hsv aug
    hsv_prob = 1.0
    # prob of applying flip aug
    flip_prob = 0.5
    # rotation angle range, for example, if set to 2, the true range is (-2, 2)
    degrees = 10.0
    # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
    translate = 0.1
    mosaic_scale = (0.1, 2)
    # apply mixup aug or not
    enable_mixup = True
    mixup_scale = (0.5, 1.5)
    # shear angle range, for example, if set to 2, the true range is (-2, 2)
    shear = 2.0

    dataset = MosaicDetection(
        dataset=dataset,
        mosaic=True,
        img_size=input_size,
        preproc=TrainTransform(
            max_labels=120,
            flip_prob=flip_prob,
            hsv_prob=hsv_prob),
        degrees=degrees,
        translate=translate,
        mosaic_scale=mosaic_scale,
        mixup_scale=mixup_scale,
        shear=shear,
        enable_mixup=enable_mixup,
        mosaic_prob=mosaic_prob,
        mixup_prob=mixup_prob,
    )
    '''
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = TrainingSampler(len(dataset))
        assert isinstance(sampler, torchdata.Sampler), f"Expect a Sampler but got {type(sampler)}"
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2**31
    seed_all_rng(initial_seed + worker_id)

def build_batch_data_loader(
    dataset,
    sampler,
    total_batch_size,
    *,
    aspect_ratio_grouping=False,
    num_workers=0,
    collate_fn=None,
    drop_last: bool = True,
    prefetch_factor=None,
    persistent_workers=False,
    pin_memory=False,
):
    """
    Build a batched dataloader. The main differences from `torch.utils.data.DataLoader` are:
    1. support aspect ratio grouping options
    2. use no "batch collation", because this is common for detection training

    Args:
        dataset (torch.utils.data.Dataset): a pytorch map-style or iterable dataset.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces indices.
            Must be provided iff. ``dataset`` is a map-style dataset.
        total_batch_size, aspect_ratio_grouping, num_workers, collate_fn: see
            :func:`build_detection_train_loader`.
        drop_last (bool): if ``True``, the dataloader will drop incomplete batches.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """

    input_size = (640, 640)
    mosaic_prob = 1.0
    # prob of applying mixup aug
    mixup_prob = 1.0
    # prob of applying hsv aug
    hsv_prob = 1.0
    # prob of applying flip aug
    flip_prob = 0.5
    # rotation angle range, for example, if set to 2, the true range is (-2, 2)
    degrees = 10.0
    # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
    translate = 0.1
    mosaic_scale = (0.1, 2)
    # apply mixup aug or not
    enable_mixup = True
    mixup_scale = (0.5, 1.5)
    # shear angle range, for example, if set to 2, the true range is (-2, 2)
    shear = 2.0
    
    dataset = MixupDetection(
        dataset=dataset,
        mosaic=True,
        img_size=input_size,
        preproc=TrainTransform(
            max_labels=120,
            flip_prob=flip_prob,
            hsv_prob=hsv_prob),
        degrees=degrees,
        translate=translate,
        mosaic_scale=mosaic_scale,
        mixup_scale=mixup_scale,
        shear=shear,
        enable_mixup=enable_mixup,
        mosaic_prob=mosaic_prob,
        mixup_prob=mixup_prob,
    )
    


    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )
    batch_size = total_batch_size // world_size

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        #dataset = ToIterableDataset(dataset, sampler, shard_chunk_size=batch_size)
        dataset = ToIterableDataset(dataset, sampler)

    #print("the dataset is : ")
    #print(dataset[0:10])

    if aspect_ratio_grouping:
        assert drop_last, "Aspect ratio grouping will drop incomplete batches."
        data_loader = torchdata.DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
        )  # yield individual mapped dict
        data_loader = AspectRatioGroupedDataset(data_loader, batch_size)
        if collate_fn is None:
            return data_loader
        return MapDataset(data_loader, collate_fn)
    else:
        return torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=num_workers,
            collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
            worker_init_fn=worker_init_reset_seed,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
        )

