# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer （siamese rpn++才有这个层）
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head(siamese mask里面才有这一层)
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def template(self, z):          #这里跟踪的时候，不考虑模板更新，将模板分支与搜索区域分支的前向分开，这里只做模板区域的分支的更新
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):             #这里跟踪的时候，不考虑模板更新，将模板分支与搜索区域分支的前向分开，这里只做搜索区域的分支的更新
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()       #softmax只能在第0个维度上执行，交换通道
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training  对应到特征图上每个anchor的信息： , , overlap（正样本和所有anchor的IOU）
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()                #cls（此anchor是正样本：1、负样本：0、忽略：-1
        label_loc = data['label_loc'].cuda()                #delta（正样本框相对于anchor的编码偏移量
        label_loc_weight = data['label_loc_weight'].cuda()  #正样本对应的那些anchor的权重，其他位置为0

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.MASK.MASK:               #siamese mask
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:           #siamese rpn++
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)        #rpn相关计算

        # get loss
        cls = self.log_softmax(cls)             #softmax之后在log,将【0,1】之间的概率拉到【-inf,0】之间，后面紧接着的应该使用nlloss,  其中softmax+log+nllloss 等价于CrossEntropyLoss,这里之所以要拆解开的原因是我们需要按照anchor的mask来计算损失
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss
        return outputs
