import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """
    Focal loss for highly imbalanced binary data (like piano rolls).
    It down-weights the massive amount of easy '0's (silence) and forces 
    the model to focus on predicting the rare '1's (actual notes).
    """
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha  # Weight for the positive class (active notes)
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Note: inputs are RAW LOGITS. Do not pass them through sigmoid before this function.
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate probabilities to compute the focal term
        pt = torch.exp(-bce_loss) 
        
        # Focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss