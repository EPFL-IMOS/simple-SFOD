"""
Dataset registration.
"""

import os
import re
from typing import Iterable

from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances, register_pascal_voc
from detectron2.data.datasets.pascal_voc import CLASS_NAMES


_root = os.getenv("DETECTRON2_DATASETS", "/scratch/username/datasets")


def register_all_datasets(cfg) -> None:
    for d in (cfg.DATASETS.TRAIN, cfg.DATASETS.TEST, cfg.DATASETS.TRAIN_TARGET):
        # if isinstance(d, dict):
        #     for dd in d.values():
        #         register_datasets(dd)
        # else:
        register_datasets(d)


def register_sim10k_voc(root_dir, year, split):
    dataset_name = f"sim10k_{year}_{split}"
    image_dir = os.path.join(root_dir, "JPEGImages")
    annotation_file = os.path.join(root_dir, "Annotations", f"{split}.xml")

    register_pascal_voc(
        name=dataset_name,
        dirname=root_dir,
        split=split,
        year=year,
        image_dir=image_dir,
        annotation_file=annotation_file,
    )


def register_datasets(datasets: Iterable[str]) -> None:
    for dataset in datasets:
        if dataset in MetadataCatalog.keys():
            continue
        # Foggy Cityscapes
        elif dataset.startswith("cityscapes_instancesonly_foggy_"):
            matched = re.match(r"cityscapes_instancesonly_foggy_(.*)_(.*)", dataset)
            if matched is None:
                raise ValueError(f"Error parsing dataset {dataset}.")
            split, fog = matched.groups()
            base_path = os.path.join(_root, "cityscapes_foggy")
            print("split is !!!!")
            print(split)
            if split == "train_foggy_beta":
                # register_coco_instances(dataset, {}, os.path.join(base_path, "annotations", f"instancesonly_filtered_gtFine_{split}_{fog}_adabn_pred.json"),
                #     base_path)
                register_coco_instances(dataset, {}, os.path.join(base_path, "annotations", f"instancesonly_filtered_gtFine_{split}_{fog}.json"),
                    base_path)
                print("register correct")
            else:
                print("a bad one")
                register_coco_instances(dataset, {}, os.path.join(base_path, "annotations", f"instancesonly_filtered_gtFine_{split}_{fog}.json"),
                    base_path)
        # Cityscapes
        elif dataset.startswith("cityscapes_instancesonly"):
            matched = re.match(r"cityscapes_instancesonly_(.*)", dataset)
            if matched is None:
                raise ValueError(f"Error parsing dataset {dataset}.")
            split = matched.groups()[0]
            base_path = os.path.join(_root, "cityscapes")
            register_coco_instances(dataset, {}, os.path.join(base_path, "annotations", f"instancesonly_filtered_gtFine_{split}.json"),
                base_path)
        # Clipart, Comic and Watercolor
        elif dataset.startswith("clipart") or dataset.startswith("comic") or dataset.startswith("watercolor"):
            matched = re.match(r"(.*)_(.*)", dataset)
            if matched is None:
                raise ValueError(f"Error parsing dataset {dataset}.")
            dataset_name, split = matched.groups()
            base_path = os.path.join(_root, dataset_name)
            class_names = CLASS_NAMES if dataset_name == "clipart" else ["bicycle", "bird", "car", "cat", "dog", "person"]
            register_pascal_voc(dataset, base_path, split, year=2012, class_names=class_names)
            if dataset_name == "clipart":
                MetadataCatalog.get(dataset).evaluator_type = "pascal_voc"
            else:
                MetadataCatalog.get(dataset).evaluator_type = "pascal_voc_6classes"
        # Sim10k
        elif dataset.startswith("sim10k"):
            matched = re.match(r"(.*)_(.*)", dataset)
            if matched is None:
                raise ValueError(f"Error parsing dataset {dataset}.")
            dataset_name, split = matched.groups()
            base_path = os.path.join(_root, dataset_name)
            register_coco_instances(dataset, {}, os.path.join(base_path, f"sim10k_coco_{split}.json"),
                base_path)
        # kitti
        elif dataset.startswith("kitti"):
            matched = re.match(r"(.*)_(.*)", dataset)
            if matched is None:
                raise ValueError(f"Error parsing dataset {dataset}.")
            dataset_name, split = matched.groups()
            base_path = os.path.join(_root, dataset_name)
            register_coco_instances(dataset, {}, os.path.join(base_path, f"kitti_{split}_coco_format.json"),
                base_path)
        else:
            ValueError(f"Dataset {dataset} is not supported.")
