"""
Base trainer class.
"""
import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase, hooks
from detectron2.engine.train_loop import AMPTrainer
from detectron2.evaluation import DatasetEvaluators, COCOEvaluator, PascalVOCDetectionEvaluator
from detectron2.data import MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.events import EventStorage
from detectron2.layers import FrozenBatchNorm2d
import logging

from daod.engine.hooks import ValLossHook
from daod.evaluation import F1Evaluator, NewCOCOEvaluator, DECE, SimCOCOEvaluator
from daod.data import DatasetMapperAnnotation


class BaseTrainer(DefaultTrainer):
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
        #return build_detection_train_loader(cfg, mapper=DatasetMapperAnnotationMixUp(cfg, is_train=True))

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
    # test_results_initial = trainer.test(cfg, model)
    # print(test_results_initial)
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
            print(i)
            data = next(trainer._trainer._data_loader_iter)
            #print(data)
            if (data):
                # print(data)
                # print("here")
                with torch.no_grad():
                    result = model(data)
            else:
                break
            del data
            del result
            torch.cuda.empty_cache()  

            if i > 1400:
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


def reset_bn_stats(module):
    """Reset running statistics in the BatchNorm layers."""
    if isinstance(module, nn.BatchNorm2d):
        print(module)
        module.running_mean = nn.Parameter(torch.zeros_like(module.running_mean), requires_grad=False)
        module.running_var = nn.Parameter(torch.ones_like(module.running_var), requires_grad=False)

def recursive_traversal(module):
    for child in module.children():
        reset_bn_stats(child)
        recursive_traversal(child)

def adabn_refinement(cfg, model, evaluators=None):
    recursive_traversal(model)
    recursive_traversal(model)
    # replace all the batch statistics

    # we will use the target - training data
    # calculate the statistics for the full datasets
    return test_refinement(cfg, model)

