import copy
import numpy as np
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import Boxes, pairwise_iou
from detectron2.modeling.postprocessing import detector_postprocess


class F1Evaluator(DatasetEvaluator):
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
            #print("input: ")
            #print(input["instances"])
            #print("output: ")
            #print(output["instances"])
            prediction = {"image_id": input["image_id"]}
            prediction["input_instances"] = input["instances"]

            if "instances" in output:
                prediction["instances"] = detector_postprocess(copy.deepcopy(output["instances"]), *prediction["input_instances"].image_size)
                # prediction["instances"] = output["instances"]
            #print("after postprocess: ")
            #print(prediction["instances"])
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def find_ious(self, eval_boxes, output_boxes):
        """
        Both are np.array of boxes in form of x1, y1, x2, y2
        """
        eval_areas = (eval_boxes[:, 2] - eval_boxes[:, 0] + 1) * (eval_boxes[:, 3] - eval_boxes[:, 1] + 1)
        output_areas = (output_boxes[:, 2] - output_boxes[:, 0] + 1) * (output_boxes[:, 3] - output_boxes[:, 1] + 1)
        # The array of IoUs
        ious = np.zeros((len(eval_boxes), len(output_boxes)))
        # calculate the IOU
        # for item in eval_boxes:
        #     if (item[0] > 600 or item[1] > 600 or item[2] > 600 or item[3] > 600):
        #         print("The eval boxes are not resized correctly") 
        # for item in output_boxes:
        #     if (item[0] > 600 or item[1] > 600 or item[2] > 600 or item[3] > 600):
        #         print("The output boxes are not resized correctly")
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
        ious = self.find_ious(eval_boxes, output_boxes)
        #     import pdb
        #     pdb.set_trace()
        result = {"true_positive": 0, "false_negative": 0, "false_positive": 0, "true_negative": 0}
        # ious conditions
        eval_trues = []
        output_trues = []
        while True:
            ret = np.where((ious > self.iou) & (ious == ious.max()))
            if len(ret[0]) > 0:
                # TODO: Only take the first one is not very suitable, take the one that is max and min with others (strict)
                eval_true_idx = ret[0][0]
                output_true_idx = ret[1][0]
                ious[eval_true_idx, :] = 0  # clean the rows
                ious[:, output_true_idx] = 0  # clean the columns
                eval_trues.append(eval_true_idx)
                output_trues.append(output_true_idx)
            else:
                break
        result["true_positive"] = len(eval_trues)
        # False positives => we predicted in but not in
        result["false_positive"] = sum([1 for i in range(len(output_boxes)) if i not in output_trues])
        # False negatives => in eval but not in true eval
        result["false_negative"] = sum([1 for i in range(len(eval_boxes)) if i not in eval_trues])
        return result

    def evaluate_output(self, prediction):

        image_id = prediction["image_id"]
        instances = prediction["instances"]

        output_scores = instances.scores.to('cpu').data.numpy()
        output_boxes = np.array([box.cpu().numpy() for box in instances.pred_boxes])
        # print(output_boxes)
        output_classes = instances.pred_classes.to('cpu').data.numpy()
        
        # Filtering the outputs
        if len(output_boxes) > 0:
            # filter by scores #TODO: Different classes may have different thresh
            keep = np.where(output_scores >= self.score)[0]
            output_boxes = output_boxes[keep]
            output_classes = output_classes[keep]
            output_scores = output_scores[keep]
            
            # take only top_n
            keep = np.argsort(output_scores)[::-1][:self.top_n]
            output_boxes = output_boxes[keep]
            output_classes = output_classes[keep]
            ouput_scores = output_scores[keep]
            output_boxes = output_boxes.astype(np.int32)

        eval_boxes = np.array([box.cpu().numpy() for box in prediction["input_instances"].gt_boxes])
        eval_classes = prediction["input_instances"].gt_classes.to('cpu').data.numpy()

        result = [{'true_positive': 0, 'false_negative': 0, 'false_positive': 0, 'true_negative': 0} for _ in range(self.class_number)]

        for category_id in range(self.class_number):
            eval_keep_for_cls = np.where(eval_classes == category_id)[0]
            output_keep_for_cls = np.where(output_classes == category_id)[0]
            if len(eval_keep_for_cls) == 0:  # there are no evals but saying there are => false positive
                result[category_id]['false_positive'] = len(output_keep_for_cls)
            if len(output_keep_for_cls) == 0:  # there are evals but there are no predictions => false negative
                result[category_id]['false_negative'] = len(eval_keep_for_cls)
            if len(eval_keep_for_cls) > 0 and len(output_keep_for_cls) > 0:  # There are both we need to check
                eval_boxes_for_cls = eval_boxes[eval_keep_for_cls]
                output_boxes_for_cls = output_boxes[output_keep_for_cls]
                result[category_id] = self.count_confusions(eval_boxes_for_cls, output_boxes_for_cls)
        
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
        # print(all_confusions)
        '''
        true_positives = sum(
            [sum([confusion[cls_idx]['true_positive'] for cls_idx in range(self.class_number)]) for confusion in all_confusions])
        false_positives = sum(
            [sum([confusion[cls_idx]['false_positive'] for cls_idx in range(self.class_number)]) for confusion in all_confusions])
        false_negatives = sum(
            [sum([confusion[cls_idx]['false_negative'] for cls_idx in range(self.class_number)]) for confusion in all_confusions])
        '''
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for confusion in all_confusions:
            for cls_idx in range(self.class_number):
                if confusion[cls_idx] is None:
                    continue
                else:
                    true_positives += confusion[cls_idx]['true_positive']
                    false_positives += confusion[cls_idx]['false_positive']
                    false_negatives += confusion[cls_idx]['false_negative']

        if (true_positives + false_positives == 0):
            precision = 0
        else:
            precision = true_positives / (true_positives + false_positives)
        if (true_positives + false_negatives == 0):
            recall = 0
        else:
            recall = true_positives / (true_positives + false_negatives)
        # print("precision")
        # print(precision)
        # print("recall")
        # print(recall)
        if (precision + recall == 0):
            return {'F1 Score': 0}
        # print(2 * (precision * recall) / (precision + recall))
        #self.storage.put_scalar(
        #    "calibration/f1", 2 * (precision * recall) / (precision + recall)
        #)
        return {'F1 Score': 2 * (precision * recall) / (precision + recall)}
