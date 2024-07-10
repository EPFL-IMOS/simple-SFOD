import copy
import numpy as np
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import Boxes, pairwise_iou
from detectron2.modeling.postprocessing import detector_postprocess
import torch.nn as nn


class TcdLoss(nn.Module):
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """
    def __init__(
        self,
        class_number,
        iou = 0.5,
        top_n = 5,
        score = 0.5
    ):
        super(nn.Module, self).__init__()
        self.class_number = class_number
        self.iou = iou
        self.top_n = top_n
        self.score = score
        self._predictions = []

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        self._predictions = []

    def process(self, inputs, outputs):
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
        for input, output in zip(inputs, outputs):

            prediction["input_instances"] = input["instances"]

            prediction["instances"] = output["instances"]
              

            if len(prediction) > 1:
                self._predictions.append(prediction)

    def find_ious(self, eval_boxes, output_boxes):
        """
        Both are np.array of boxes in form of x1, y1, x2, y2
        """
        eval_areas = (eval_boxes[:, 2] - eval_boxes[:, 0] + 1) * (eval_boxes[:, 3] - eval_boxes[:, 1] + 1)
        output_areas = (output_boxes[:, 2] - output_boxes[:, 0] + 1) * (output_boxes[:, 3] - output_boxes[:, 1] + 1)
        # The array of IoUs
        ious = torch.zeros((len(eval_boxes), len(output_boxes))).cuda()

        for eval_idx in range(len(eval_boxes)):
            xx1 = torch.max(eval_boxes[eval_idx, 0], output_boxes[:, 0])
            yy1 = torch.max(eval_boxes[eval_idx, 1], output_boxes[:, 1])
            xx2 = torch.min(eval_boxes[eval_idx, 2], output_boxes[:, 2])
            yy2 = torch.min(eval_boxes[eval_idx, 3], output_boxes[:, 3])
            w = torch.max(0.0, xx2 - xx1 + 1)
            h = torch.max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (eval_areas[eval_idx] + output_areas - inter)
            ious[eval_idx, :] = ovr

        return ious
    
    def count_confusions(self, eval_boxes, output_boxes, iou_thresh=0.5):
        """
        Inputs must be in np.array xyxy format
        """
        ious = self.find_ious(eval_boxes, output_boxes)


        #ret_tp = np.where((ious > self.iou) & (ious == ious.max()))
        #ret_fp = np.where((ious <= self.iou) & (ious == ious.max()))

        ious_mask = (ious > iou_thresh) & (ious == ious.max(dim=1, keepdim=True).values)
        ret_tp = torch.nonzero(ious_mask, as_tuple=True)
        print(ret_tp)

        return (ret_tp, ret_fp)


    def evaluate_output(self, prediction):

        instances = prediction["instances"]

        output_scores = instances.scores
        output_boxes = instances.pred_boxes
        output_classes = instances.pred_classes

        eval_boxes = prediction["input_instances"].gt_boxes
        eval_classes = prediction["input_instances"].gt_classes

        result = {}

        for category_id in range(self.class_number):
            eval_keep_for_cls = np.where(eval_classes == category_id)[0]
            output_keep_for_cls = np.where(output_classes == category_id)[0]
            if len(eval_keep_for_cls) > 0 and len(output_keep_for_cls) > 0:  # There are both we need to check
                eval_boxes_for_cls = eval_boxes[eval_keep_for_cls]
                output_boxes_for_cls = output_boxes[output_keep_for_cls]
                result[category_id] = self.count_confusions(eval_boxes_for_cls, output_boxes_for_cls)
            else:
                result[category_id] = None

        return result

    

    def forward(self, inputs, outputs):
        self.process(inputs, outputs)
        
        m = nn.Tanh()
        loss_PC_l = []

        for prediction in self._predictions:
            result = self.evaluate_output(prediction)

            for cls_idx in range(self.class_number):
                if result_conf[cls_idx] is None:
                    continue
                else:
                    truep, falsep = result_conf[cls_idx]
                    TP_l = truep
                    FP_l = falsep
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

                    if denom > 0.0:
                        detPC = torch.log(1+(numr.cuda()/denom.cuda()))
                        loss_PC_l.append(detPC)

        if loss_PC_l:
            detPC_mean = torch.mean(torch.stack(loss_PC_l))
        else:
            detPC_mean = torch.tensor(0.0).cuda()

        return detPC_mean           

