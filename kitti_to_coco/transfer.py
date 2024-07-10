import os
import json
import cv2
import numpy as np
from collections import defaultdict

kitti_base_path = "/cluster/scratch/username/kitti"
kitti_split = "train"  # or "val"
img_dir = os.path.join(kitti_base_path, "train_images")
ann_dir = os.path.join(kitti_base_path, "labels_9")

coco_output = {
    "info": {},
    "licenses": [],
    "categories": [{"id": 1, "name": "car"}, 
        {"id": 2, "name": "train"}, 
        {"id": 3, "name": "person"}, 
        {"id": 4, "name": "bicycle"}, 
        {"id": 5, "name": "rider"}, 
        {"id": 6, "name": "motorcycle"}, 
        {"id": 7, "name": "bus"}, 
        {"id": 8, "name": "truck"}],
    "images": [],
    "annotations": []
}
KITTI_TO_COCO_CATEGORY_MAP = {
    "Car": 1,
    "Pedestrian": 3,
    "Cyclist": 4,
}
annotation_id = 0


for idx, ann_file in enumerate(os.listdir(ann_dir)):
    if ann_file.endswith(".txt"):
        img_id = int(ann_file.split(".")[0])
        img_path = os.path.join(img_dir, f"{img_id:06d}.png")
        height, width = cv2.imread(img_path).shape[:2]
        #print(img_path)
        #print(os.path.basename(img_path))
        img_dict = {
            "id": img_id,
            "file_name": img_path,
            "width": width,
            "height": height,
        }
        coco_output["images"].append(img_dict)

        with open(os.path.join(ann_dir, ann_file)) as f:
            for line in f:
                data = line.strip().split(" ")
                obj_class, truncated, occluded, alpha, x1, y1, x2, y2, h, w, l, x, y, z, yaw = data

                if obj_class in KITTI_TO_COCO_CATEGORY_MAP:
                    category_id = KITTI_TO_COCO_CATEGORY_MAP[obj_class]

                    ann_dict = {
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": category_id,
                        "bbox": [float(x1), float(y1), float(x2) - float(x1), float(y2) - float(y1)],
                        "area": (float(x2) - float(x1)) * (float(y2) - float(y1)),
                        "iscrowd": 0,
                    }
                    coco_output["annotations"].append(ann_dict)
                    annotation_id += 1


with open(os.path.join(kitti_base_path, f"kitti_{kitti_split}_coco_format.json"), "w") as outfile:
    json.dump(coco_output, outfile)
