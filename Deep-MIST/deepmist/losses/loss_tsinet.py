import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target):
    smooth = 0.00

    intersection = pred * target

    intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
    pred_sum = torch.sum(pred, dim=(1, 2, 3))
    target_sum = torch.sum(target, dim=(1, 2, 3))
    loss = (intersection_sum + smooth) / \
           (pred_sum + target_sum - intersection_sum + smooth)

    loss = 1 - torch.mean(loss)
    return loss


class MTWHLoss(nn.Module):
    """Loss used to train TSINet (ported from TSINet/losses.py).

    The original forward applies sigmoid before calling this loss;
    here the sigmoid is applied internally so it follows the Deep-MIST
    convention of passing raw logits.
    """

    def __init__(self, threshold=0.4):
        super(MTWHLoss, self).__init__()
        self.threshold = threshold

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        # 计算目标像素和背景像素的占比
        gt = (targets > self.threshold).float()
        pt = (inputs > self.threshold).float()
        Mt = (targets > 0.5).float()
        loss_dice = dice_loss(inputs, Mt)
        pos_ratio = torch.mean(Mt)
        neg_ratio = 1 - pos_ratio

        squared_difference = torch.abs(pt - gt)
        loss_ou = torch.sum(squared_difference) / gt.size(0)
        weight = neg_ratio / torch.maximum(pos_ratio, torch.tensor(1e-6, device=pos_ratio.device))
        # weighted BCELoss (equivalent to nn.BCELoss(weight=weight), mean reduction)
        inputs = inputs.clamp(1e-6, 1 - 1e-6)
        loss1 = -(weight * Mt * torch.log(inputs) + (1 - Mt) * torch.log(1 - inputs)).mean()

        loss = loss1 + loss_dice * 10 + 0.2 * loss_ou
        return loss
