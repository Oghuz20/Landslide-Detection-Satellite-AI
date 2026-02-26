"""
WeightedBCEDiceLoss — BCE with pos_weight + Dice loss.
pos_weight=12.0 handles severe class imbalance (~1.9% landslide pixels)
and forces the model toward higher recall.
"""

import torch
import torch.nn as nn


class WeightedBCEDiceLoss(nn.Module):

    def __init__(self, pos_weight=12.0, dice_w=0.7, smooth=1.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])
        self.dice_w     = dice_w
        self.bce_w      = 1.0 - dice_w
        self.smooth     = smooth

    def forward(self, pred, target):
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        if self.pos_weight.device != pred.device:
            self.pos_weight = self.pos_weight.to(pred.device)

        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            pred, target, pos_weight=self.pos_weight, reduction='mean'
        )

        prob         = torch.sigmoid(pred).reshape(-1)
        tgt          = target.reshape(-1)
        intersection = (prob * tgt).sum()
        dice_loss    = 1 - (2 * intersection + self.smooth) / \
                           (prob.sum() + tgt.sum() + self.smooth)

        return self.bce_w * bce_loss + self.dice_w * dice_loss