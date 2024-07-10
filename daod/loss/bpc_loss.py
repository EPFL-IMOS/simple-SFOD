import copy
import numpy as np
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import Boxes, pairwise_iou
from detectron2.modeling.postprocessing import detector_postprocess
import torch.nn as nn
import torch


def bpc_loss(
    class_number,
    inputs,
    outputs,
    iou = 0.5,
    top_n = 5,
    score = 0.5
):
    _predictions = process(inputs, outputs)
    loss = loss_forward(_predictions, class_number, iou)
    return loss




def process(inputs, outputs):
    """
    Process the pair of inputs and outputs.
    If they contain batches, the pairs can be consumed one-by-one using `zip`:

    .. code-block:: python

        for input_, output in zip(inputs, outputs):
            # do evaluation on single input/output pair
            ...

    Args:
        inputs (list): the inputs that's used to call the model. GT
        outputs (list): the return value of `model(inputs)`. Pred
    """

    _predictions = []

    for input, output in zip(inputs, outputs):

        # print("input")
        # print(input)
        # print("output")
        # print(output)

        prediction = {}

        prediction["input_instances"] = input

        prediction["instances"] = output
            

        if len(prediction) > 1:
            _predictions.append(prediction)
    
    return _predictions

def find_ious(eval_boxes, output_boxes):
    """
    Both are np.array of boxes in form of x1, y1, x2, y2
    """
    eval_boxes = eval_boxes.tensor
    output_boxes = output_boxes.tensor
    eval_areas = (eval_boxes[:, 2] - eval_boxes[:, 0] + 1) * (eval_boxes[:, 3] - eval_boxes[:, 1] + 1)
    output_areas = (output_boxes[:, 2] - output_boxes[:, 0] + 1) * (output_boxes[:, 3] - output_boxes[:, 1] + 1)
    # The array of IoUs
    ious = torch.zeros((len(eval_boxes), len(output_boxes))).cuda()

    for eval_idx in range(len(eval_boxes)):
        xx1 = torch.max(eval_boxes[eval_idx, 0], output_boxes[:, 0])
        yy1 = torch.max(eval_boxes[eval_idx, 1], output_boxes[:, 1])
        xx2 = torch.min(eval_boxes[eval_idx, 2], output_boxes[:, 2])
        yy2 = torch.min(eval_boxes[eval_idx, 3], output_boxes[:, 3])
        w = torch.max(torch.tensor(0.0).cuda(), xx2 - xx1 + 1)
        h = torch.max(torch.tensor(0.0).cuda(), yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (eval_areas[eval_idx] + output_areas - inter)
        ious[eval_idx, :] = ovr

    return ious

def count_confusions(eval_boxes, output_boxes, iou_thresh=0.5):
    """
    Inputs must be in np.array xyxy format
    """

    #print("number of gt: ")
    #print(len(eval_boxes))
    #print("number of predictions")
    #print(len(output_boxes))

    ious = find_ious(eval_boxes, output_boxes)

    #print("ios shape: ")
    #print(ious.shape)


    #ret_tp = np.where((ious > self.iou) & (ious == ious.max()))
    #ret_fp = np.where((ious <= self.iou) & (ious == ious.max()))
    #print("ious")
    #print(ious)
    ious_tp_mask = (ious > iou_thresh) & (ious == ious.max(dim=0, keepdim=True).values)
    #print("ious tp mask")
    #print(ious_tp_mask)
    #print("where output")
    #print(np.where(ious_tp_mask.cpu().numpy() == 1))
    ious_tp_map = torch.Tensor((np.where(ious_tp_mask.cpu().numpy() == 1)[1])).cuda()
    ious_tp_map = ious_tp_map.long()
    #print("ious tp map")
    #print(ious_tp_map)

    # ret_tp = ious[ious_tp_mask]
    # print("ret tp")
    # print(ret_tp)

    # ious_fp_mask = (ious <= iou_thresh) & (ious == ious.max(dim=0, keepdim=True).values)
    # ious_fp_map = torch.Tensor((np.where(ious_fp_mask.cpu().numpy() == 1)[0])).cuda()
    # ious_fp_map = ious_fp_map.long()
    ious_fp_mask = torch.ones((len(output_boxes)), dtype=torch.bool)
    for item in ious_tp_map:
        ious_fp_mask[item] = False
    ious_fp_map = torch.arange(len(output_boxes)) 
    ious_fp_map = ious_fp_map[ious_fp_mask]  
    #print("ious_fp_map")
    #print(ious_fp_map)

    #print("--------------------------")
    #print(len(ious_tp_map))
    #print(len(ious_fp_map))
    return (ious_tp_map, ious_fp_map)


def evaluate_output(prediction, class_number, iou_thresh):

    instances = prediction["instances"]
    #print(instances)

    #objectness_logits = instances.objectness_logits
    #print("objective logits")
    #print(objectness_logits)
    #print("objective logits shape: ")
    #print(objectness_logits.shape)
    output_scores = instances.scores
    #print("output scores")
    #print(output_scores)
    #print("output scores shape: ")
    #print(output_scores.shape)
    output_boxes = instances.pred_boxes
    #print("proposal boxes shape: ")
    #print(len(output_boxes)) 

    output_classes = instances.pred_classes
    #print(len(output_classes))
    #print(output_classes)

    eval_boxes = prediction["input_instances"].gt_boxes
    eval_classes = prediction["input_instances"].gt_classes
    #eval_boxes = instances.gt_boxes
    #eval_classes = instances.gt_classes
    #print(len(eval_classes))
    #print(eval_classes)


    result = {}
    scores = {}

    for category_id in range(class_number):
        eval_keep_for_cls = (eval_classes == category_id)
        #print(eval_keep_for_cls)
        # eval_valid_map = torch.Tensor((np.where(eval_keep_for_cls.cpu().numpy() == 1)[0])).cuda()
        # eval_valid_map = eval_valid_map.long()
        # eval_keep_for_cls = eval_valid_map
        # eval_keep_for_cls = np.where(eval_classes == category_id)[0]

        output_keep_for_cls = (output_classes == category_id)
        #print(output_keep_for_cls)
        # output_valid_map = torch.Tensor((np.where(output_keep_for_cls.cpu().numpy() == 1)[0])).cuda()
        # output_valid_map = output_valid_map.long()        
        # output_keep_for_cls = output_valid_map
        # output_keep_for_cls = np.where(output_classes == category_id)[0]

        if len(eval_boxes[eval_keep_for_cls]) == 0:  # there are no evals but saying there are => false positive
            #print("here here here")
            result[category_id] = (torch.tensor([0]).cuda(), output_scores[output_keep_for_cls])
            #print(output_scores[output_keep_for_cls])
        elif len(eval_boxes[eval_keep_for_cls]) > 0 and len(output_boxes[output_keep_for_cls]) > 0:  # There are both we need to check
            eval_boxes_for_cls = eval_boxes[eval_keep_for_cls]
            output_boxes_for_cls = output_boxes[output_keep_for_cls]
            ious_tp_map, ious_fp_map = count_confusions(eval_boxes_for_cls, output_boxes_for_cls, iou_thresh)
            
            output_scores_for_cls = output_scores[output_keep_for_cls]
            result[category_id] = (output_scores_for_cls[ious_tp_map], output_scores_for_cls[ious_fp_map])
        else:
            result[category_id] = None

    return result



def loss_forward(_predictions, class_number, iou_thresh):
    
    m = nn.Tanh()
    loss_PC_l = []

    for prediction in _predictions:
        result = evaluate_output(prediction, class_number, iou_thresh)
        nAC = 0
        nAN = 0
        nIC = 0
        nIN = 0
        for cls_idx in range(class_number):
            if result[cls_idx] is None:
                continue
            else:
                truep, falsep = result[cls_idx]
                TP_l = truep
                FP_l = falsep
                #print("TP_l")
                #print(TP_l)
                AC = TP_l[TP_l>=0.5] * m(TP_l[TP_l>=0.5])
                AN = TP_l[TP_l<0.5] * (1-m(TP_l[TP_l<0.5]))
                #print("AN")
                #print(AN)
                IC = (1-FP_l[FP_l>=0.5]) * m(FP_l[FP_l>=0.5])
                IN = (1-FP_l[FP_l<0.5]) * (1-m(FP_l[FP_l<0.5]))
                
                nAC += AC.sum()
                nAN += AN.sum()
                nIC += IC.sum()
                nIN += IN.sum()

        #print("nAC: ")
        #print(nAC)
        #print("nAN: ")
        #print(nAN)
        #print("nIC: ")
        #print(nIC)
        #print("nIN: ")
        #print(nIN)

        numr = nAN+nIC
        denom = nAC+nIN
        #denom = nAC

        if denom > 0.0:
            detPC = torch.log(1+(numr.cuda()/denom.cuda()))
            loss_PC_l.append(detPC)
    if loss_PC_l:
        detPC_mean = torch.mean(torch.stack(loss_PC_l))
    else:
        detPC_mean = torch.tensor(0.0).cuda()

    #print("******************************")
    #print("The BPC loss is: ")
    #print(loss_PC_l)
    #print(detPC_mean)


    return detPC_mean           