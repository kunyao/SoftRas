import torch
import torch.nn as nn
import numpy as np


def iou(predict, target, eps=1e-6):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + eps
    return (intersect / union).sum() / intersect.nelement()

def iou_loss(predict, target):
    return 1 - iou(predict, target)

def multiview_iou_loss(predicts, targets_a, targets_b):
    # target_a = targets_a[:, 3].clone()
    # target_a = targets_a.clone()
    # target_a[target_a >= 0.3] = 1.0
    # target_a[target_a < 0.3] = 0.0

    # target_b = targets_b[:, 3].clone()
    # target_b = targets_b.clone()
    # target_b[target_b >= 0.3] = 1.0
    # target_b[target_b < 0.3] = 0.0

    loss0 = iou_loss(predicts[0][:, 3], targets_a)
    loss1 = iou_loss(predicts[1][:, 3], targets_a)
    loss2 = iou_loss(predicts[2][:, 3], targets_b)
    loss3 = iou_loss(predicts[3][:, 3], targets_b)

    # hard0 = predicts[0].clone()
    # hard0[hard0 >= 0.95] = 1.0
    # hard0[hard0 < 1.0] = 0.0
    # loss4 = iou_loss(hard0[:, 3], targets_a[:, 3])
    # loss4 = iou_loss(hard0[:, 3], targets_a)

    # hard1 = predicts[1].clone()
    # hard1[hard1 >= 0.95] = 1.0
    # hard1[hard1 < 1.0] = 0.0
    # loss5 = iou_loss(hard1[:, 3], targets_a[:, 3])
    # loss5 = iou_loss(hard1[:, 3], targets_a)

    return loss0, loss1, loss2, loss3

def multiview_rgb_loss(predicts, targets_a, targets_b):

    mse = nn.MSELoss()
    loss0 = mse(predicts[0][:, :3], targets_a[:, :3])
    loss1 = mse(predicts[1][:, :3], targets_a[:, :3])
    loss2 = mse(predicts[2][:, :3], targets_b[:, :3])
    loss3 = mse(predicts[3][:, :3], targets_b[:, :3])

    return (loss0 + loss1 + loss2 + loss3) / 4
