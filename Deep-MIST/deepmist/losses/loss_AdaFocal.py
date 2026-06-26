import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        area_weight = self._get_area_weight(target)
        smooth = 1

        pred_sigmoid = pred.sigmoid()
        intersection = pred_sigmoid * target
        iou = (intersection.sum() + smooth) / (pred_sigmoid.sum() + target.sum() - intersection.sum() + smooth)
        iou = torch.clamp(iou, min=1e-6, max=1 - 1e-6).detach()
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        target = target.type(torch.long)
        at = target * area_weight + (1 - target) * (1 - area_weight)
        pt = torch.exp(-bce_loss)
        pt = torch.clamp(pt, min=1e-6, max=1 - 1e-6)
        focal_loss = (1 - pt) ** (1 - iou + 1e-6) * bce_loss

        focal_loss = at * focal_loss
        return focal_loss.sum()

    def _get_area_weight(self, target):
        area = target.sum(dim=(1, 2, 3))
        return torch.sigmoid(1 - area / (area.max() + 1)).view(-1, 1, 1, 1)

    def adafocal_gradient(self, iou, weight, x):
        sigmoid_x = 1 / (1 + np.exp(-x))
        term1 = weight * sigmoid_x * (1 - iou) * (1 - sigmoid_x) ** (1 - iou) * np.log(sigmoid_x)
        term2 = weight * (1 - sigmoid_x) ** (2 - iou)
        return term1 - term2
