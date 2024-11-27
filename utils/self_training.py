import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceBasedSelfTrainingLoss(nn.Module):
    """
    Self training loss that adopts confidence threshold to select reliable pseudo labels from
    `Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks (ICML 2013)
    <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.664.3543&rep=rep1&type=pdf>`_.
    Args:
        threshold (float): Confidence threshold.
    Inputs:
        - y: unnormalized classifier predictions.
        - y_target: unnormalized classifier predictions which will used for generating pseudo labels.
    Returns:
         A tuple, including
            - self_training_loss: self training loss with pseudo labels.
            - mask: binary mask that indicates which samples are retained (whose confidence is above the threshold).
            - pseudo_labels: generated pseudo labels.
    Shape:
        - y, y_target: :math:`(minibatch, C)` where C means the number of classes.
        - self_training_loss: scalar.
        - mask, pseudo_labels :math:`(minibatch, )`.
    """

    def __init__(self, threshold: float):
        super(ConfidenceBasedSelfTrainingLoss, self).__init__()
        self.threshold = threshold

    def forward(self, y, y_target):
        confidence, pseudo_labels = F.softmax(y_target.detach(), dim=1).max(dim=1)
        mask = (confidence >= self.threshold).float()
        self_training_loss = (F.cross_entropy(y, pseudo_labels, reduction='none') * mask).mean()

        return self_training_loss, mask, pseudo_labels, confidence


class AdaptiveConfidenceBasedSelfTrainingLoss(nn.Module):

    def __init__(self, threshold: float, num_classes: int):
        super(AdaptiveConfidenceBasedSelfTrainingLoss, self).__init__()
        self.threshold = threshold
        self.num_classes = num_classes
        self.classwise_acc = torch.ones((self.num_classes,)).cuda()

    def update(self, selected_labels):
        """Update dynamic per-class accuracy."""
        if selected_labels.nelement() > 0:
            sigma = selected_labels.bincount(minlength=self.num_classes)
            self.classwise_acc = sigma / sigma.max()

    def forward(self, y, y_target):
        confidence, pseudo_labels = F.softmax(y_target.detach(), dim=1).max(dim=1)

        # mask = (confidence >= self.threshold * (self.classwise_acc[pseudo_labels] + 1.) / 2).float()  # linear
        # mask = (confidence >= self.threshold * (1 / (2. - self.classwise_acc[pseudo_labels]))).float()  # low_limit
        mask = (confidence >= self.threshold * (self.classwise_acc[pseudo_labels] / (2. - self.classwise_acc[pseudo_labels]))).float()  # convex
        # mask = (confidence >= self.threshold * (torch.log(self.classwise_acc[pseudo_labels] + 1.) + 0.5)/(math.log(2) + 0.5)).float()  # concave

        self_training_loss = (F.cross_entropy(y, pseudo_labels, reduction='none') * mask).mean()

        return self_training_loss, mask, pseudo_labels, confidence


class MCDUncertaintyBasedSelfTrainingLoss(nn.Module):
    """
    Self training loss that adopts MCD uncertainty threshold to select reliable pseudo labels.
    Args:
        threshold (float): Confidence threshold.
    Inputs:
        - y: unnormalized classifier predictions.
        - y_target: MCD inferences with unnormalized classifier predictions which will used for generating pseudo labels.
        - reduction: aggregation of softmax scores across MCD inferences to estimate uncertainty.
            - mean
            - variance
            - entropy
    Returns:
         A tuple, including
            - self_training_loss: self training loss with pseudo labels.
            - mask: binary mask that indicates which samples are retained (whose confidence is above the threshold).
            - pseudo_labels: generated pseudo labels.
    Shape:
        - y: :math:`(minibatch, C)` where C means the number of classes.
        - y_target: :math:`(minibatch, N, C)` where N means the number of MCD inferences.
        - self_training_loss: scalar.
        - mask, pseudo_labels :math:`(minibatch, )`.
    """

    def __init__(self, threshold: float):
        super(MCDUncertaintyBasedSelfTrainingLoss, self).__init__()
        self.threshold = threshold

    def forward(self, y, y_target):
        confidence, pseudo_labels = F.softmax(y_target.detach(), dim=2).mean(axis=1).max(dim=1)
        mask = (confidence >= self.threshold).float()
        self_training_loss = (F.cross_entropy(y, pseudo_labels, reduction='none') * mask).mean()

        return self_training_loss, mask, pseudo_labels, confidence


class OracleSelfTrainingLoss(nn.Module):
    """
    Self training loss that uses the target labels to generate pseudo labels.
    Inputs:
        - y: unnormalized classifier predictions.
        - y_target: unnormalized classifier predictions which will used for generating pseudo labels.
        - y_true: true target labels.
    Returns:
         A tuple, including
            - self_training_loss: self training loss with pseudo labels.
            - mask: binary mask that indicates which samples are retained (the correct ones).
            - pseudo_labels: generated pseudo labels.
    Shape:
        - y, y_target: :math:`(minibatch, C)` where C means the number of classes.
        - y_true: :math:`(minibatch, )`.
        - self_training_loss: scalar.
        - mask, pseudo_labels :math:`(minibatch, )`.
    """

    def __init__(self):
        super(OracleSelfTrainingLoss, self).__init__()

    def forward(self, y, y_target, y_true):
        confidence, pseudo_labels = F.softmax(y_target.detach(), dim=1).max(dim=1)
        mask = (pseudo_labels == y_true).float()
        self_training_loss = (F.cross_entropy(y, pseudo_labels, reduction='none') * mask).mean()

        return self_training_loss, mask, pseudo_labels, confidence
