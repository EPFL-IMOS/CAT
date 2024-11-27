import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """
    Focal loss.
    Args:
        gamma: Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                'none': No reduction will be applied to the output.
                'mean': The output will be averaged.
                'sum': The output will be summed.
    """
    def __init__(
        self,
        weight=None,
        gamma: float = 2,
        reduction: str = "none",
    ):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Focal loss.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the
                    classification label for each element in inputs.
        Returns:
            Loss tensor with the reduction option applied.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p = torch.exp(-ce_loss)
        loss = ce_loss * ((1 - p) ** self.gamma)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
