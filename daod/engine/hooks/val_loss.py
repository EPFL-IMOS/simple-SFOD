# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.engine.hooks import HookBase
import detectron2.utils.comm as comm

import torch


class ValLossHook(HookBase):
    def __init__(self, eval_period, model, data_loader, model_name=""):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self._model_name = model_name

    def _do_loss_eval(self):
        record_acc_dict = {}
        with torch.no_grad():
            for _, inputs in enumerate(self._data_loader):
                record_dict = self._get_loss(inputs, self._model)
                # accumulate the losses
                for loss_type in record_dict.keys():
                    if loss_type not in record_acc_dict.keys():
                        record_acc_dict[loss_type] = record_dict[loss_type]
                    else:
                        record_acc_dict[loss_type] += record_dict[loss_type]
            # average
            for loss_type in record_acc_dict.keys():
                record_acc_dict[loss_type] = record_acc_dict[loss_type] / len(
                    self._data_loader
                )

            # divide loss and other metrics
            loss_acc_dict = {}
            for key in record_acc_dict.keys():
                if key[:4] == "loss":
                    loss_acc_dict[key] = record_acc_dict[key]

            self._write_losses(loss_acc_dict)

    def _get_loss(self, data, model):

        record_dict = model(data)

        #print("Record Dict: ")
        #print(record_dict)
        # Some models such as teacher-student return a tuple
        if (isinstance(record_dict, tuple)):
            record_dict = record_dict[0]
        
        #print("Record Dict After: ")
        #print(record_dict)

        if isinstance(record_dict, list):
            return {}

        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in record_dict.items()
        }

        #print("metrics dict: ")
        #print(metrics_dict)
        return metrics_dict

    def _write_losses(self, metrics_dict):
        # only output the results of major node
        if comm.is_main_process():
            total_losses_reduced = sum(metrics_dict.values())

            self.trainer.storage.put_scalar(
                "total_loss" + self._model_name + "_val", total_losses_reduced
            )

            metrics_dict = {
                k + self._model_name + "_val": metrics_dict[k]
                for k in metrics_dict.keys()
            }

            if len(metrics_dict) > 1:
                self.trainer.storage.put_scalars(**metrics_dict)

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.trainer.iter, loss_dict
                )
            )

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
