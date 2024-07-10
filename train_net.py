"""
Main script to train DAOD models.

"""
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from daod.config import add_config
from daod.data.datasets import register_all_datasets
from daod.engine.trainers import (
    BaseTrainer,
    BaseWQTrainer,
    BaseMosaicTrainer,
    DATrainer,
    AdaptiveTeacherTrainer,
    SourceFreeAdaptiveTeacherTrainer,
    BaseMixupTrainer,
    BaseMosaicWQTrainer,
    BaseMosaicWQNewTrainer,
    SourceFreeAdaptiveTeacherSingleTrainer
)

# To register the architectures
from daod.modeling.meta_arch import *
from daod.modeling.proposal_generator import *
from daod.modeling.roi_heads import *
from daod.engine.trainers import base

import torch
torch.multiprocessing.set_sharing_strategy('file_system')


def setup(args):
    """Setup config."""
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    """Main script."""
    cfg = setup(args)
    if cfg.TRAINER == "base":
        Trainer = BaseTrainer
    elif cfg.TRAINER == "base_wq":
        Trainer = BaseWQTrainer
    elif cfg.TRAINER == "base_mosaic":
        Trainer = BaseMosaicTrainer
    elif cfg.TRAINER == "base_mosaic_wq":
        Trainer = BaseMosaicWQTrainer
    elif cfg.TRAINER == "base_mosaic_wq_new":
        Trainer = BaseMosaicWQNewTrainer
    elif cfg.TRAINER == "base_mixup":
        Trainer = BaseMixupTrainer
    elif cfg.TRAINER == "source_free_adaptive_teacher_single":
        Trainer = SourceFreeAdaptiveTeacherSingleTrainer
    elif cfg.TRAINER == "da":
        Trainer = DATrainer
    elif cfg.TRAINER == "adaptive_teacher":
        Trainer = AdaptiveTeacherTrainer
    elif cfg.TRAINER == "source_free_adaptive_teacher":
        Trainer = SourceFreeAdaptiveTeacherTrainer
    else:
        raise ValueError(f"Trainer {cfg.TRAINER} not found.")

    register_all_datasets(cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        #return Trainer.test(cfg, model)
        # return Trainer.test_refinement(cfg, model)
        return base.test_refinement(cfg, model)

    trainer = Trainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command line args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
