import json

# Load the predictions
#with open('/cluster/scratch/username/weights/adabn_weight2/inference/coco_instances_results.json', 'r') as file:
#    predictions = json.load(file)

# with open('/cluster/scratch/username/weights/source_prediction/inference/coco_instances_results.json', 'r') as file:
with open('/cluster/scratch/username/cityscape_detectron_output/r_101_c4_cs_adabn/inference/coco_instances_results.json', 'r') as file:
    predictions = json.load(file)

# with open('/cluster/scratch/username/cityscapes_foggy/annotations/instancesonly_filtered_gtFine_train_foggy_beta_0.02_original.json', 'r') as file:
with open('//scratch/izar/likhobab/datasets/cityscapes_foggy/annotations/instancesonly_filtered_gtFine_train_foggy_beta_0.02.json', 'r') as file:
    dics = json.load(file)

#for key in dics:
#    print(key)

# Convert predictions to ground truth format
ground_truths = []
idd = 1
for prediction in predictions:
    if prediction["score"] < 0.7:
        continue
    # Assuming each prediction contains 'bbox' and 'category_id'
    ground_truth = {
        "image_id": prediction["image_id"],
        "bbox": prediction["bbox"],  # [x,y,width,height]
        "category_id": prediction["category_id"],
        "id": idd
    }
    idd += 1
    ground_truths.append(ground_truth)

dics["annotations"] = ground_truths

# Assume we're converting to COCO format for the ground truth
#coco_format = {
#    "images": [],  # This would be filled with image info
#    "annotations": ground_truths,
#    "categories": [],  # This would be filled with category info
#}

# Save the ground truth annotations
with open('//scratch/izar/likhobab/instancesonly_filtered_gtFine_train_foggy_beta_0.02.json', 'w') as file:
    json.dump(dics, file, indent=4)
