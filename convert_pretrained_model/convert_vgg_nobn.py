# modified from detectron2 conversion script
import pickle as pkl
import sys
import torch
import numpy as np

'''
Usage:
  # download  vgg16 models without BN from torchvision, I only check vgg16 without BN for my research
  wget $URL_IN_TROCHVISION -O vgg16.pth
  # run the conversion
  ./convert-torchvision-to-d2.py vgg16.pth vgg16.pkl


here is vgg16 weights

vgg_block1
features.0.weight
features.0.bias
features.2.weight
features.2.bias

vgg_block2
features.5.weight
features.5.bias
features.7.weight
features.7.bias

vgg_block3
features.10.weight
features.10.bias
features.12.weight
features.12.bias
features.14.weight
features.14.bias

vgg_block4
features.17.weight
features.17.bias
features.19.weight
features.19.bias
features.21.weight
features.21.bias

vgg_block5
features.24.weight
features.24.bias
features.26.weight
features.26.bias
features.28.weight
features.28.bias

do not care
classifier.0.weight
classifier.0.bias
classifier.3.weight
classifier.3.bias
classifier.6.weight
classifier.6.bias

to

backbone.vgg_block1.0.conv1.{bias, weight} 
backbone.vgg_block1.0.conv2.{bias, weight} 
backbone.vgg_block2.0.conv1.{bias, weight} 
backbone.vgg_block2.0.conv2.{bias, weight} 
backbone.vgg_block3.0.conv1.{bias, weight} 
backbone.vgg_block3.0.conv2.{bias, weight} 
backbone.vgg_block3.0.conv3.{bias, weight} 
backbone.vgg_block4.0.conv1.{bias, weight} 
backbone.vgg_block4.0.conv2.{bias, weight} 
backbone.vgg_block4.0.conv3.{bias, weight} 
backbone.vgg_block5.0.conv1.{bias, weight} 
backbone.vgg_block5.0.conv2.{bias, weight} 
backbone.vgg_block5.0.conv3.{bias, weight} 


proposal_generator.rpn_head.anchor_deltas.{bias, weight} 
proposal_generator.rpn_head.conv.{bias, weight} 
proposal_generator.rpn_head.objectness_logits.{bias, weight} 
roi_heads.box_predictor.bbox_pred.{bias, weight} 
roi_heads.box_predictor.cls_score.{bias, weight}
'''


if __name__ == "__main__":
    # input = sys.argv[1]
    input_model = "/cluster/scratch/username/simple-SFOD/vgg16-nobn.pth"
    output_model = "/cluster/scratch/username/simple-SFOD/vgg16-nobn.pkl"

    obj = torch.load(input_model, map_location="cpu")
    #old_v = 0
    vgg_block_id = 0
    conv_ind = 1
    newmodel = {}
    for k in list(obj.keys()):
        print(k)
    
    print("---------------------------------------")

    # construct transfer array
    transfer_array_block = np.zeros(26)
    transfer_array_conv = np.zeros(26)
    old_block_id = 0
    conv_id_index = 0
    index_array = np.array([[0, 0, 2, 2, 5, 5], [0, 0, 3, 3, 5, 5], [0, 0, 3, 3, 5, 5, 7, 7], [0, 0, 2, 2, 4, 4]])
    for i in range(26):
        block_number = i//2
        # print(block_number)
        vgg_block_id = 0
        if block_number == 0 or block_number == 1 or block_number == 2:
            vgg_block_id = 0
        elif block_number == 3 or block_number == 4 or block_number == 5:
            vgg_block_id = 1
        elif block_number == 6 or block_number == 7 or block_number == 8 or block_number == 9:
            vgg_block_id = 2
        elif block_number == 10 or block_number == 11 or block_number == 12:
            vgg_block_id = 3

        transfer_array_block[i] = vgg_block_id

        if vgg_block_id != old_block_id:
            conv_id_index = 0
            old_block_id = vgg_block_id
        
        transfer_array_conv[i] = index_array[vgg_block_id][conv_id_index]
        conv_id_index += 1

    
    index = 0
    for k in list(obj.keys()):
        #index += 1

        parse = k.split('.')
        if parse[0] == 'classifier':
            break
        #v = int(parse[1])
        #if v - old_v > 2:
        #    vgg_block_id += 1
        #    conv_ind = 0
        # new_k = f'vgg_block{vgg_block_id}.0.conv{conv_ind}.{parse[-1]}' 
        #print(transfer_array_block[index])
        new_k = f'backbone.vgg{int(transfer_array_block[index])}.{int(transfer_array_conv[index])}.{parse[-1]}' 
        #if parse[-1] == 'weight':
        #    conv_ind += 1
        #if parse[-1] == 'running_var':
        #    conv_ind = 0      
        
        #old_v = v
        index += 1
        print(k, "->", new_k)
        newmodel[new_k] = obj.pop(k).detach().numpy()

    for k in list(newmodel.keys()):
        print(k)

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True} 
    with open(output_model, "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())

# There will be a matching 
# 6 layers is a block
# (2 + 2 + 3 + 3 + 3) * 6 = 13 * 6 = 78
# [2, 2, 3, 3, 3]
# [0-1, 2-3, 4-6, 7-9, 10-12]
# [vgg0.0-4, vgg1.0-4, vgg2.0-7, vgg3.0-7, vgg4.0-7]

'''
Some model parameters or buffers are not found in the checkpoint:
[34mbackbone.vgg0.0.{bias, weight}[0m
[34mbackbone.vgg0.2.{bias, weight}[0m
[34mbackbone.vgg0.5.{bias, weight}[0m
[34mbackbone.vgg1.0.{bias, weight}[0m
[34mbackbone.vgg1.3.{bias, weight}[0m
[34mbackbone.vgg1.5.{bias, weight}[0m
[34mbackbone.vgg2.0.{bias, weight}[0m
[34mbackbone.vgg2.3.{bias, weight}[0m
[34mbackbone.vgg2.5.{bias, weight}[0m
[34mbackbone.vgg2.7.{bias, weight}[0m
[34mbackbone.vgg3.0.{bias, weight}[0m
[34mbackbone.vgg3.2.{bias, weight}[0m
[34mbackbone.vgg3.4.{bias, weight}[0m
[34mproposal_generator.rpn_head.anchor_deltas.{bias, weight}[0m
[34mproposal_generator.rpn_head.conv.{bias, weight}[0m
[34mproposal_generator.rpn_head.objectness_logits.{bias, weight}[0m
[34mroi_heads.box_head.fc1.{bias, weight}[0m
[34mroi_heads.box_head.fc2.{bias, weight}[0m
[34mroi_heads.box_predictor.bbox_pred.{bias, weight}[0m
[34mroi_heads.box_predictor.cls_score.{bias, weight}[0m
[5m[31mWARNING[0m [32m[03/25 22:41:40 fvcore.common.checkpoint]: [0mThe checkpoint state_dict contains keys that are not used by the model:
  [35mfeatures.0.{bias, weight}[0m
  [35mfeatures.2.{bias, weight}[0m
  [35mfeatures.5.{bias, weight}[0m
  [35mfeatures.7.{bias, weight}[0m
  [35mfeatures.10.{bias, weight}[0m
  [35mfeatures.12.{bias, weight}[0m
  [35mfeatures.14.{bias, weight}[0m
  [35mfeatures.17.{bias, weight}[0m
  [35mfeatures.19.{bias, weight}[0m
  [35mfeatures.21.{bias, weight}[0m
  [35mfeatures.24.{bias, weight}[0m
  [35mfeatures.26.{bias, weight}[0m
  [35mfeatures.28.{bias, weight}[0m
  [35mclassifier.0.{bias, weight}[0m
  [35mclassifier.3.{bias, weight}[0m
  [35mclassifier.6.{bias, weight}[0m
'''