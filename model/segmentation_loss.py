#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce, repeat


class BinaryDiceLoss(nn.Module):
    """
    Dice loss of binary class
    
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1e-7, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        """
        predict: A tensor of shape [B, C, H, W], value 0~1
        target: A binary tensor of shape same with predict
        """
        assert predict.shape == target.shape, f'predict {predict.shape} & target {target.shape} do not match'
        
        predict = rearrange(predict.contiguous(), 'b c h w -> (b c) (h w)')   # B*C, H*W*D 
        target = rearrange(target.contiguous(), 'b c h w -> (b c) (h w)')

        intersection = 2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        union = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
        
        loss = 1 - intersection / union

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        
class SampleWiseBinaryDiceLoss(nn.Module):
    """
    sample-wise Dice Loss
    """
    def __init__(self, smooth=1e-7, p=2, reduction='mean'):
        super(SampleWiseBinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        """
        predict: A tensor of shape [C, H, W], value 0~1
        target: A binary tensor of shape same with predict
        """
        assert predict.shape == target.shape, f'predict {predict.shape} & target {target.shape} do not match'
        
        predict = rearrange(predict.contiguous(), 'c h w -> c (h w)')   # B*C, H*W
        target = rearrange(target.contiguous(), 'c h w -> c (h w)')

        intersection = 2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        union = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - intersection / union

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
        
class SegmentationLoss(nn.Module):
    """
    Weighted Dice and BCE Loss
    
    Args:
        dice_loss: Loss calculator for Dice loss
        bce_w_logits_loss: Loss calculator for BCE with logits loss
        weight (float): Weight for the combined loss
    """
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.dice_loss = BinaryDiceLoss(reduction='none')
        self.bce_w_logits_loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, logits, mask, query_mask=None):
        """
        Calculate Weighted Dice and BCE Loss

        Args:
            logits (tensor): unsigmoided prediction, bnhw
            mask (tensor): binary, bnhw
            query_mask (tensor): binary, bn, indicating which query is padding
        Returns:
            tuple: (combined_loss, unreduced_batch_dice_loss, unreduced_batch_ce_loss)
        """
        if query_mask is None:
            query_mask = torch.ones((mask.shape[0], mask.shape[1]))
        
        prediction = torch.sigmoid(logits) 
        batch_dice_loss = self.dice_loss(prediction, mask)   # (b*n)
        batch_dice_loss = rearrange(batch_dice_loss, '(b c) -> b c', b=prediction.shape[0]) # b n
        batch_dice_loss = batch_dice_loss * query_mask
        reduced_batch_dice_loss = torch.sum(batch_dice_loss) / (torch.sum(query_mask) + 1e-14)  # bn -> 1
        unreduced_batch_dice_loss = torch.sum(batch_dice_loss, dim=1) / (torch.sum(query_mask, dim=1) + 1e-14).detach()  # bn -> b

        batch_ce_loss = self.bce_w_logits_loss(logits, mask)  # (b, n, h, w)
        batch_ce_loss = torch.mean(batch_ce_loss, dim=(2,3)) # b n
        batch_ce_loss = batch_ce_loss * query_mask
        reduced_batch_ce_loss = torch.sum(batch_ce_loss) / (torch.sum(query_mask) + 1e-14)  # bn -> 1
        unreduced_batch_ce_loss = torch.sum(batch_ce_loss, dim=1) / (torch.sum(query_mask, dim=1) + 1e-14).detach()  # bn -> b
        
        return torch.mean(reduced_batch_ce_loss + reduced_batch_dice_loss), unreduced_batch_dice_loss, unreduced_batch_ce_loss

if __name__ == '__main__':
    a = torch.rand((8, 3, 256, 256, 96))
    b = torch.rand((8, 3, 256, 256, 96))
    mask = torch.zeros((8, 3))
    mask[0, 1] = 1.0
    mask[1, 0] = 1.0
    
    dice_loss = BinaryDiceLoss(reduction='none')
    ce_loss = nn.BCELoss(reduction='none')
    
    dice = dice_loss(a, b)
    print(dice)
    dice = dice_loss(a, b, mask)
    print(dice)
    
    ce = torch.mean(ce_loss(a, b), dim=(2,3,4))
    print(ce)
    ce = ce * mask
    print(ce)
