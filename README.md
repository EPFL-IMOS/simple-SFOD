<div align='center'>

# Simplifying Source-Free Domain Adaptation for Object Detection: Effective Self-Training Strategies and Performance Insights

[Yan Hao](https://honeyhaoyan.github.io)<sup>1</sup>&nbsp;&nbsp;&nbsp;
[Florent Forest](https://florentfo.rest)<sup>1</sup>&nbsp;&nbsp;&nbsp;
[Olga Fink](https://people.epfl.ch/olga.fink)<sup>1</sup>
<br/>
<sub>
<sup>1</sup> Intelligent Maintenance and Operations Systems (IMOS), EPFL, Lausanne, Switzerland
</sub>

ECCV, 2024

[![Paper](https://img.shields.io/badge/paper-ECCV-174c80)](https://link.springer.com/chapter/10.1007/978-3-031-72949-2_12)&nbsp;&nbsp;&nbsp;[![Arxiv](https://img.shields.io/badge/arXiv-2407.07586-B31B1B)](http://arxiv.org/abs/2407.07586)

</div>

Source code for the implementation of the paper [Simplifying Source-Free Domain Adaptation for Object Detection: Effective Self-Training Strategies and Performance Insights](https://link.springer.com/chapter/10.1007/978-3-031-72949-2_12).

## Requirements

* pytorch >= 1.12
* detectron2 0.6

## Project structure

```
simple-SFOD/
├── cityscapes-to-coco-conversion/: Cityscapes dataset preparation tool
├── sim10k-to-coco/: Sim10k dataset preparation tool
├── kitti-to-coco/: KITTI dataset preparation tool
├── configs/: configuration files for detectron2 models
├── convert_pretrained_model/: convert the pytorch pretrained model to detectron2 format
├── daod/
    ├── data/: submodule for data management
        ├── mappers/: submodule for detectron2 dataset mappers
        ├── transforms/: submodule for data transformations
    ├── engine/
        ├── hooks/: submodule for training hooks
        └── trainers/: submodule for trainer classes
    ├── evaluation/: submodule for evaluators
    └── modeling/: submodule for neural net architectures
        ├── meta_arch/: detectron2 meta-architectures
        ├── proposal_generator/: RPN architectures
        └── roi_heads/: RoI heads architectures
    └── config.py: set-up custom configuration
├── README.md: this readme file
├── train_net_mt.py: adaptation script for teacher-student
└── train_net.py: main script for training and evaluation
```

## Datasets

The following datasets are supported:

* detectron2 built-in datasets
* Cityscapes (instances-only)
* Foggy Cityscapes (instances-only, $\beta \in \{0.005, 0.01, 0.02\}$)
* Sim10k
* KITTI

To define the root directory containing the datasets, set the environment variable `DETECTRON2_DATASETS`, e.g. `export DETECTRON2_DATASETS=/path/to/datasets/`. If not defined, it will be assumed that datasets are in `datasets/` inside the project directory.

## Dataset preparation and structure

### Cityscapes and Foggy Cityscapes (instances-only)

First, the cityscapes and foggy cityscapes need to be downloaded from the [Cityscapes website](https://www.cityscapes-dataset.com/downloads/). Then, extract the archives and modify the folder names to obtain the following structures:

```
cityscapes
├── gtFine
│   ├── test
│   ├── train
│   └── val
├── gtFine_trainvaltest
│   ├── license.txt
│   └── README
└── leftImg8bit
    ├── test
    ├── train
    └── val
```

```
cityscapes_foggy
└── leftImg8bit
    ├── test
    ├── train
    └── val
```

The conversion of cityscapes and foggy cityscapes into COCO JSON format for object detection (instances only) is performed using the [cityscapes-to-coco-conversion tool](https://github.com/TillBeemelmanns/cityscapes-to-coco-conversion/). It has been slightly modified to remove segmentations and allow to use a suffix to handle fog level (beta). Run following commands to perform the conversion:

```shell
python cityscapes-to-coco-conversion/main.py --dataset cityscapes --datadir /path/to/cityscapes --outdir /path/to/cityscapes/annotations
```

For foggy cityscapes, run this command with the desired "beta" value:

```shell
python cityscapes-to-coco-conversion/main.py --dataset cityscapes --datadir /path/to/cityscapes --outdir /path/to/cityscapes_foggy/annotations --file_name_suffix _foggy_beta_0.02
```

A new directory `annotations` will be created, containing JSON files with instance annotations for each split (train, val, test).

### Sim10k -- TO DO
### KITTI -- TO DO
## Getting started

Run `python train_net.py --help` for a usage instructions.

### Train Source Model

For cityscapes:

```shell
python train_net.py --num-gpus 1 --config-file configs/faster_rcnn_VGG_cityscapes_source_new.yaml
```
### Adaptation
For cityscapes to cityscapes foggy:

With teacher-student, use the pre-trained model to initialize weights for both teacher and student. Then run:

```shell
python train_net_mt.py --num-gpus 1 --config-file configs/faster_rcnn_VGG_cityscapes_foggy_adaptive_teacher_source_free.yaml
```

With fixed pseudo-labels, firstly get the AdaBN model, then use it to initialize weights. Run:

```shell
python train_net.py --num-gpus 1 --config-file configs/faster_rcnn_VGG_cityscapes_foggy_source_wq.yaml
```

## Citation

If this work was useful to you, please cite our paper:

```BibTeX
@inproceedings{hao_simplifying_2024,
    title = {Simplifying Source-Free Domain Adaptation for Object Detection: Effective Self-Training Strategies and Performance Insights},
    author = {Hao, Yan and Forest, Florent and Fink, Olga},
    booktitle = {ECCV 2024},
    year = {2024},
}
```
