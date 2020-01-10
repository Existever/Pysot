# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F
import numpy as np


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



def weight_feat_loss(pred_feat, label_feat, bbox,stride=21.0):
    '''
    :param pred_feat: gru预测的特征map,shape为 【b,c,h,w]
    :param label_feat: gru下一帧的模板提取的的特征map,shape为 【b,c,h,w]
    :param bbox:目标在模板图下的bbox,对应模板图分支输入分辨率
    :param stride: 输入模板图到特征对应stride=127/6=21.0
    :return:
    '''

    fb,fc,fh,fw=pred_feat.shape

    loss = (pred_feat - label_feat).abs()                                   #对于 feature map都直接算1范数
    return loss.sum().div(fb*fc*fw*fh)                       #除以batch平均一下



    # fb,fc,fh,fw=pred_feat.shape
    # bbox =bbox/stride
    # x = torch.arange(fw).reshape(1,1,fw).repeat(fb,fh, 1).float()
    # y = torch.arange(fh).reshape(1,fh,1).repeat(fb,1,fw).float()
    # bbox = bbox.reshape(fb,1,1,4).round().float()                                   #对坐标四舍五入
    # x1, y1, x2, y2 = bbox[...,0],bbox[...,1],bbox[...,2],bbox[...,3]
    #
    #
    # cond = (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
    # mask=torch.where(cond, torch.FloatTensor([0.9]),torch.FloatTensor([0.1])).cuda()  #满足条件的目标区域的损失占比为0.9，背景区域占比为0.1
    # mask= mask.reshape(fb,-1,fh,fw)
    # diff = (pred_feat - label_feat).abs()                                   #对于 feature map都直接算1范数
    # diff = diff.sum(dim=1).view(fb, -1, fh, fw)
    # loss = diff * mask                              #乘以权重相当于只在正样本位置求平均
    # return loss.sum().div(fb*fc*fw*fh)                       #除以batch平均一下

