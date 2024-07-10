import torch
import torch.nn as nn
import torch.nn.functional as F
from daod.src.evaluators.coco_evaluator import get_coco_metrics
import numpy as np
from daod.src.bounding_box import BoundingBox
from daod.src.utils.enumerators import BBFormat, BBType, CoordinatesType

def construct_bbox_list(predictions, type):
    image_index = 0
    predictions_construct = []
    for image in predictions:
        image_name = str(image_index)
        print(image)
        # print(image.fields)
        bbox_length = len(image.get(type + '_boxes'))
        for i in range(bbox_length):
            class_id = image.get(type + '_classes')[i]
            print(class_id)
            coordinate = image.get(type + '_boxes')[i]
            coordinate = coordinate[0]
            print(coordinate)
            bb_format = BBFormat.XYX2Y2
            confidence = image.get('scores')[i]
            print(confidence)
            predictions_construct.append(BoundingBox(image_name = image_name, class_id = class_id, coordinates = coordinate, confidence = confidence, format = bb_format))
            i += 1

        image_index += 1
    
    return predictions_constructs

def tcd_loss(predictions, gt):
    print("predictions")
    print(predictions)
    print("gt instances")
    print(gt)
    print("--------------------------------")
            # Categorize all the dets into precision space
    
    predictions_construct = construct_bbox_list(predictions, 'pred')
    gt_construct = construct_bbox_list(gt, 'gt')


    """ Constructor.

        Parameters
        ----------
            image_name : str
                String representing the name of the image.
            class_id : str
                String value representing class id.
            coordinates : tuple
                Tuple with 4 elements whose values (float) represent coordinates of the bounding \\
                    box.
                The coordinates can be (x, y, w, h)=>(float,float,float,float) or(x1, y1, x2, y2)\\
                    =>(float,float,float,float).
                See parameter `format`.
            type_coordinates : Enum (optional)
                Enum representing if the bounding box coordinates (x,y,w,h) are absolute or relative
                to size of the image. Default:'Absolute'.
            img_size : tuple (optional)
                Image size in the format (width, height)=>(int, int) representinh the size of the
                image of the bounding box. If type_coordinates is 'Relative', img_size is required.
            bb_type : Enum (optional)
                Enum identifying if the bounding box is a ground truth or a detection. If it is a
                detection, the confidence must be informed.
            confidence : float (optional)
                Value representing the confidence of the detected object. If detectionType is
                Detection, confidence needs to be informed.
            format : Enum
                Enum (BBFormat.XYWH or BBFormat.XYX2Y2) indicating the format of the coordinates of
                the bounding boxes.
                BBFormat.XYWH: <left> <top> <width> <height>
                BBFomat.XYX2Y2: <left> <top> <right> <bottom>.
                BBFomat.YOLO: <x_center> <y_center> <width> <height>. (relative)
        """

    
    fin_res = get_coco_metrics(gt_construct, predictions_construct, iou_threshold=0.5, area_range=(0, np.inf), max_dets=100)

    loss_PC_l = []

    if not fin_res:
        detPC = torch.tensor(0.0).cuda()

    for di,g in fin_res.items():
        if g['TP'] is not None and g['FP'] is not None:
            TP_l = g['conf_scr'][:g['TP']]
            FP_l = g['conf_scr'][g['TP']:g['FP']+g['TP']]
            m = nn.Tanh()

            AC = TP_l[TP_l>=0.5] * m(TP_l[TP_l>=0.5])
            AN = TP_l[TP_l<0.5] * (1-m(TP_l[TP_l<0.5]))
            IC = (1-FP_l[FP_l>=0.5]) * m(FP_l[FP_l>=0.5])
            IN = (1-FP_l[FP_l<0.5]) * (1-m(FP_l[FP_l<0.5]))

            nAC = AC.sum()
            nAN = AN.sum()
            nIC = IC.sum()
            nIN = IN.sum()

            numr = nAN+nIC
            denom = nAC+nIN

            print("nAC: ")
            print(nAC)
            print("nAN: ")
            print(nAN)
            print("nIC: ")
            print(nIC)
            print("nIN: ")
            print(nIN)


            if denom>0.0:
                detPC = torch.log(1+(numr.cuda()/denom.cuda()))
                loss_PC_l.append(detPC)


    if loss_PC_l:
        detPC_mean = torch.mean(torch.stack(loss_PC_l))
    else:
        detPC_mean = torch.tensor(0.0).cuda()

    return detPC_mean