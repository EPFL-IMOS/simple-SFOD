import copy
import numpy as np
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import Boxes, pairwise_iou
from detectron2.modeling.postprocessing import detector_postprocess
import numpy as np
from netcal.scaling import LogisticCalibration, LogisticCalibrationDependent
from netcal.metrics import ECE

class DECE(DatasetEvaluator):
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
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        for input, output in zip(inputs, outputs):
            # print(input, output)
            prediction = {"image_id": input["image_id"]}
            prediction["input_instances"] = input["instances"]

            if "instances" in output:
                prediction["instances"] = detector_postprocess(copy.deepcopy(output["instances"]), *prediction["input_instances"].image_size)
                # prediction["instances"] = output["instances"]

            if len(prediction) > 1:
                self._predictions.append(prediction)

    def find_ious(self, eval_boxes, output_boxes):
        """
        Both are np.array of boxes in form of x1, y1, x2, y2
        """
        #print(eval_boxes)
        #print(output_boxes)
        eval_areas = (eval_boxes[:, 2] - eval_boxes[:, 0] + 1) * (eval_boxes[:, 3] - eval_boxes[:, 1] + 1)
        output_areas = (output_boxes[:, 2] - output_boxes[:, 0] + 1) * (output_boxes[:, 3] - output_boxes[:, 1] + 1)
        # The array of IoUs
        ious = np.zeros((len(eval_boxes), len(output_boxes)))
        for eval_idx in range(len(eval_boxes)):
            xx1 = np.maximum(eval_boxes[eval_idx, 0], output_boxes[:, 0])
            yy1 = np.maximum(eval_boxes[eval_idx, 1], output_boxes[:, 1])
            xx2 = np.minimum(eval_boxes[eval_idx, 2], output_boxes[:, 2])
            yy2 = np.minimum(eval_boxes[eval_idx, 3], output_boxes[:, 3])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (eval_areas[eval_idx] + output_areas - inter)
            ious[eval_idx, :] = ovr
            
        # eval_boxes = Boxes(eval_boxes)
        # output_boxes = Boxes(output_boxes)
        # overlap = pairwise_iou(eval_boxes, output_boxes)
        return ious
    
    def count_confusions(self, eval_boxes, output_boxes, iou_thresh=0.5):
        """
        Inputs must be in np.array xyxy format
        """
        #print(eval_boxes)
        #print(output_boxes)
        ious = self.find_ious(eval_boxes, output_boxes)
        eval_matched = np.zeros(len(eval_boxes))
        output_matched = np.zeros(len(output_boxes))

        result = {"true_positive": 0, "false_negative": 0, "false_positive": 0, "true_negative": 0}
        
        #print("ious: ")
        #print(ious)
        while True:
            ret = np.where((ious > self.iou) & (ious == ious.max()))
            #print("the ret matrix is: ")
            #print(ret)
            if len(ret[0]) > 0:
                # TODO: Only take the first one is not very suitable, take the one that is max and min with others (strict)
                eval_true_idx = ret[0][0]
                output_true_idx = ret[1][0]
                #print(ious[eval_true_idx, :])
                #ious[eval_true_idx, :] = 0  # clean the rows
                #print(ious[eval_true_idx, :])
                ious[:, output_true_idx] = 0  # clean the columns
                #eval_matched[eval_true_idx] = 1
                output_matched[output_true_idx] = 1
            else:
                break
        
        output_matched = output_matched.astype(np.int32)
        #print(output_matched)
        return output_matched


    def ece_calculation(self, confidences, matched):
        n_bins = len(confidences)
        ece = ECE(n_bins, detection=True)
        confidences = confidences.reshape(-1, 1)
        uncalibrated_score = ece.measure(confidences, matched)
        return uncalibrated_score

    
    def evaluate_output(self, prediction):
        class_number = self.class_number

        instances = prediction["instances"]
        output_scores = instances.scores
        output_boxes = instances.pred_boxes
        output_classes = instances.pred_classes

        output_scores = output_scores.to('cpu').data.numpy()
        output_boxes = np.array([box.cpu().numpy() for box in output_boxes])
        output_classes = output_classes.to('cpu').data.numpy()

        eval_boxes = prediction["input_instances"].gt_boxes
        eval_classes = prediction["input_instances"].gt_classes

        eval_boxes = np.array([box.cpu().numpy() for box in eval_boxes])
        eval_classes = eval_classes.to('cpu').data.numpy()

        result = {}
        scores = {}

        for category_id in range(class_number):
            eval_keep_for_cls = (eval_classes == category_id)
            output_keep_for_cls = (output_classes == category_id)

            if len(eval_boxes[eval_keep_for_cls]) == 0:  # there are no evals but saying there are => false positive
                #result[category_id] = ([], output_scores[output_keep_for_cls])
                result[category_id] = (output_scores[output_keep_for_cls], np.zeros(len(output_boxes[output_keep_for_cls])))
            elif len(output_boxes[output_keep_for_cls]) == 0:
                result[category_id] = (np.array([]), np.array([]))
            elif len(eval_boxes[eval_keep_for_cls]) > 0 and len(output_boxes[output_keep_for_cls]) > 0:  # There are both we need to check
                eval_boxes_for_cls = eval_boxes[eval_keep_for_cls]
                output_boxes_for_cls = output_boxes[output_keep_for_cls]
                output_matched = self.count_confusions(eval_boxes_for_cls, output_boxes_for_cls, self.iou)
                output_scores_for_cls = output_scores[output_keep_for_cls]
                #scores = output_boxes_for_cls[output_matched]
                #print("scores: ")
                #print(output_boxes_for_cls)
                #print("matched: ")
                #print(output_matched)
                #print("len matched")
                #print(len(output_matched))
                #print("len output scores")
                #print(len(output_boxes_for_cls))
                result[category_id] = (output_scores_for_cls, output_matched)
                #output_scores_for_cls = output_scores[output_keep_for_cls]
                #result[category_id] = (output_scores_for_cls[ious_tp_map], output_scores_for_cls[ious_fp_map])
       

        return result


    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        all_confusions = [self.evaluate_output(prediction) for prediction in
                      self._predictions]


        matched = np.array([])
        confidences = np.array([])
        for item in all_confusions:
            for key in item:
                confi, mat = item[key]
                confidences = np.concatenate((confidences, confi))
                matched = np.concatenate((matched, mat))
        dece = self.ece_calculation(confidences, matched)
        
        '''
        overall_dict = {}
        number_dict = {}
        for item in all_confusions:
            # print(item)
            for key in item:
                if key in overall_dict:
                    overall_dict[key] += item[key]
                    number_dict[key] += 1
                else:
                    overall_dict[key] = item[key]
                    number_dict[key] = 1
        num = 0
        sum = 0
        for key in overall_dict:
            sum += overall_dict[key]/number_dict[key]
            num += 1
        dece = sum / num
        '''
        print("the ece loss is: ")
        print(dece)
        
        return {'DECE': dece}