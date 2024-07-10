import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveConfidenceBasedSelfTrainingLoss(nn.Module):

    def __init__(self, threshold: float, num_classes: int):
        super(AdaptiveConfidenceBasedSelfTrainingLoss, self).__init__()
        self.threshold = threshold
        self.num_classes = num_classes
        self.classwise_acc = torch.ones((self.num_classes,))
        self.classwise_acc = self.classwise_acc.cuda()

    def update(self, selected_labels):
        """Update dynamic per-class accuracy."""
        if selected_labels.nelement() > 0:
            sigma = selected_labels.bincount(minlength=self.num_classes)
            self.classwise_acc = sigma / sigma.max()

    def forward(self, confidence, pseudo_labels):
        # confidence, pseudo_labels = F.softmax(y_target.detach(), dim=1).max(dim=1)

        # mask = (confidence >= self.threshold * (self.classwise_acc[pseudo_labels] + 1.) / 2).float()  # linear
        # mask = (confidence >= self.threshold * (1 / (2. - self.classwise_acc[pseudo_labels]))).float()  # low_limit
        #print(confidence.device)
        #print(self.classwise_acc.device)
        #print(pseudo_labels.device)
        mask = (confidence >= self.threshold * (self.classwise_acc[pseudo_labels] / (2. - self.classwise_acc[pseudo_labels]))).float()  # convex
        # mask = (confidence >= self.threshold * (torch.log(self.classwise_acc[pseudo_labels] + 1.) + 0.5)/(math.log(2) + 0.5)).float()  # concave
    
        # self_training_loss = (F.cross_entropy(y, pseudo_labels, reduction='none') * mask).mean()

        return mask