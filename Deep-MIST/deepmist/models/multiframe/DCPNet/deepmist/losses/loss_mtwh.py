import numpy as np
import torch
import torch.nn as nn


def dice_loss(pred, target):
    smooth = 0.0
    intersection = pred * target
    intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
    pred_sum = torch.sum(pred, dim=(1, 2, 3))
    target_sum = torch.sum(target, dim=(1, 2, 3))
    loss = (intersection_sum + smooth) / (
        pred_sum + target_sum - intersection_sum + smooth
    )
    return 1 - torch.mean(loss)


class MTWHLoss(nn.Module):
    def __init__(self, threshold=0.4):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        gt = (target > self.threshold).float()
        pt = (pred > self.threshold).float()
        mt = (target > 0.5).float()

        loss_dice = dice_loss(pred, mt)
        pos_ratio = torch.mean(mt)
        neg_ratio = 1 - pos_ratio

        loss_ou = torch.abs(pt - gt).sum() / gt.size(0)

        weight = neg_ratio / torch.clamp(pos_ratio, min=1e-6)
        bce_loss = nn.BCELoss(weight=weight)

        loss_bce = bce_loss(pred, mt)
        loss = loss_bce + loss_dice * 10 + 0.2 * loss_ou
        return loss
