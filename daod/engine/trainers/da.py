"""
DA trainer class.
"""
import time

from daod.data import build_detection_da_train_loader
from daod.engine.trainers import BaseTrainer


class DATrainer(BaseTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_da_train_loader(cfg)  # produces tuples of (source, target) data
