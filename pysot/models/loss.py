# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F


def get_cls_loss(pred, label, select):          #依据位置索引 select计算对应位置处的交叉熵损失
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)            #NLLLoss的结果就是把log_softmax的结果与Label对应的那个值拿出来相乘，再去掉负号，再求均值


def select_cross_entropy_loss(pred, label): #按照mask位置选择计算交叉熵
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    '''
    :param pred_loc: 预测的bbox信息，shape为 【b,4*anchor_num,size,size]
    :param label_loc: gt的标签信息，shape为【b,4,anchor_num,size,size]
    :param loss_weight:
    :return:
    '''
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()         #对于 【dx,dy dw dh】都直接算1范数
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight                   #乘以权重相当于只在正样本位置求平均
    return loss.sum().div(b)                    #除以batch平均一下
