import os
import json
import xml.etree.ElementTree as ET
from collections import defaultdict

# Set your dataset root directory
dataset_root = "/cluster/scratch/username/sim10k/VOC2012/"
annotations_dir = os.path.join(dataset_root, "Annotations")

# Collect all unique object class names
class_name_to_id = defaultdict(int)
class_id_counter = 0

# Initialize COCO-format dictionary
coco_format = {
    "info": {
        "description": "Sim10k Dataset",
        "version": "1.0",
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [],
}

# Process each annotation file
annotation_id = 1
for image_id, annotation_file in enumerate(os.listdir(annotations_dir), start=1):
    if not annotation_file.endswith(".xml"):
        continue

    file_path = os.path.join(annotations_dir, annotation_file)
    tree = ET.parse(file_path)

    # Add image information to the COCO-format dictionary
    # file_name = tree.find("filename").text
    file_name = "JPEGImages/" + annotation_file[:-4] + ".jpg"
    width = int(tree.find("./size/width").text)
    height = int(tree.find("./size/height").text)
    coco_format["images"].append({
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
    })

    # Process each object in the annotation file
    for obj in tree.findall("object"):
        # Get object category name and assign a unique ID if not already assigned
        category_name = obj.find("name").text
        if category_name not in class_name_to_id:
            class_name_to_id[category_name] = class_id_counter
            class_id_counter += 1

        # Get bounding box coordinates
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        width = xmax - xmin
        height = ymax - ymin

        # Add annotation to the COCO-format dictionary

        if class_name_to_id[category_name] == 0:
            category_current = 2
        elif class_name_to_id[category_name] == 1:
            category_current = 5
        else:
            category_current = 1
        coco_format["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_current,
            "bbox": [xmin, ymin, width, height],
            "area": width * height,
            "iscrowd": 0,
        })
        annotation_id += 1

# Add category information to the COCO-format dictionary
#for category_name, category_id in class_name_to_id.items():
#    coco_format["categories"].append({
#        "id": category_id,
#        "name": category_name,
#    })
'''
coco_format["categories"].append({
        "id": 1,
        "name": "person",
    })
coco_format["categories"].append({
        "id": 2,
        "name": "car",
    })
coco_format["categories"].append({
        "id": 5,
        "name": "motorcycle",
    })
'''
coco_format["categories"] = [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}, {"id": 3, "name": "bicycle"}, {"id": 4, "name": "rider"}, {"id": 5, "name": "motorcycle"}, {"id": 6, "name": "bus"}, {"id": 7, "name": "truck"}, {"id": 8, "name": "train"}]

# Save the COCO-format dictionary as a JSON file
output_path = os.path.join(dataset_root, "sim10k_coco_format.json")
with open(output_path, "w") as output_file:
    json.dump(coco_format, output_file, indent=2)
