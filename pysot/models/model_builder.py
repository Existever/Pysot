# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss,weight_feat_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck
from pysot.models.backbone.grus import GRU_Model

from pysot.utils.anchor import Anchors
from pysot.show import draw_rect
import torchvision.utils as vutils




class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)
        #是否添加gru模块
        if cfg.GRU.USE_GRU:
            self.grus =GRU_Model(cfg.GRU.SEQ_IN,cfg.GRU.SEQ_OUT)
            if self.grus.seq_out_len !=1:
                raise ValueError("For tracking task GRU_Model.seq_out_len must be set as 1\n",
                                 "please check the value of __C.GRU.SEQ_OUT in config.py file"
                                 )
            self.zfs = [None] * self.grus.seq_in_len  # 多帧模板图z的特征f,测试前向的时候使用


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

        #---------------------tenosrboard监视用-------------------
        # hanning窗口,为tensorboard显示预测结果提前出事

        hanning = np.hanning(cfg.TRAIN.OUTPUT_SIZE)  # 生成和输出特征图大小相同的hanning窗
        window = np.outer(hanning, hanning)  # 一维度的hanning窗口通过外积得到二位hanning 窗口

        self.anchor_num = len(cfg.ANCHOR.RATIOS)*len(cfg.ANCHOR.SCALES)
        self.window = torch.from_numpy(np.tile(window.flatten(), self.anchor_num)).cuda().float()          #size*size*anchor_num展平

        #按照给定的比例因子生成一个位置的多种anchor,输出shape为【anchor_num,4】其中输出的anchor的值为[-w * 0.5, -h * 0.5, w * 0.5, h * 0.5]
        self.anchors = Anchors(cfg.ANCHOR.STRIDE,       #8
                               cfg.ANCHOR.RATIOS,       # [0.33, 0.5, 1, 2, 3]
                               cfg.ANCHOR.SCALES)
        # 生成两种类型的anchor,第一种是左上右下坐标类型的，第二种是中心点类型的，shape均为【4，anchor_num,size,size]
        self.anchors.generate_all_anchors(im_c=cfg.TRAIN.SEARCH_SIZE//2,
                                          size=cfg.TRAIN.OUTPUT_SIZE)
        self.anchors_tensor =torch.from_numpy(self.anchors.all_anchors[0]).cuda()








    def template(self, z):          #这里跟踪的时候，不考虑模板更新，将模板分支与搜索区域分支的前向分开，这里只做模板区域的分支的更新

        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf



    def gru_template(self, z,idx):                      #模板区域的分支gru融合前self.grus.seq_in_len个特征

        buf_idx=idx%self.grus.seq_in_len                #当前帧在缓存中的索引
        self.zfs[buf_idx] = self.backbone(z)

        if idx<self.grus.seq_in_len-1:                  #处在初始化阶段，只返回当前的结果
            zf= self.zfs[buf_idx]
        else:
            #否则用gru融合从当前帧算起一个前seq_in_len帧的特征
            T=self.grus.seq_in_len
            gru_zfs = [None] * T  # 多帧模板图z的特征f
            for t in range(T):
                gru_zfs[T-1-t] =self.zfs[(idx-t)%T]   #buf_t_idx=(idx-t)%T当前帧的前t帧在buf中的位置

            gru_zfs=torch.stack(gru_zfs,dim=1)            #将输入变为[n,t,c,h,w]的形式
            zf =self.grus(gru_zfs).squeeze(dim=1)          #grus输出为[n,1,c,h，w]的形式转化为【n,c,h,w】的形式



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

    def show_result(self,cls,loc,search):

        b,_,h,w =cls.shape
        score=cls.reshape(b,2,-1,h,w)
        score=F.softmax(score, dim=1)[:,1,...]      #!!!!通道1才是代表正样本(是目标)
        score = score.view(b, -1)     #anchor层面上取取概率值,-1代表的size为anchor_num*size*size
        score = (score * (1 - 0.4) + self.window.view(1,-1) * 0.4)   #hanning窗口加权
        pos = torch.argmax(score, dim=1).view(-1)                    #batch的数据也展开
        index = torch.arange(pos.shape[0], device=pos.device)        #bath的索引

        #self.anchors_tensor中心点类型的，shape均为【4，anchor_num,size,size]
        loc1d = loc.view(b, 4, -1, h,w)

        XYWH= torch.stack(
            [(loc1d[:, 0, ...] * self.anchors_tensor[2, ...] + self.anchors_tensor[0, ...]).view(loc1d.shape[0], -1)[index, pos],
             (loc1d[:, 1, ...] * self.anchors_tensor[3, ...] + self.anchors_tensor[1, ...]).view(loc1d.shape[0], -1)[index, pos],
             (torch.exp(loc1d[:, 2, ...]) * self.anchors_tensor[2, ...]).view(loc1d.shape[0], -1)[index, pos],
             (torch.exp(loc1d[:, 3, ...]) * self.anchors_tensor[3, ...]).view(loc1d.shape[0], -1)[index, pos]],
            dim=1)


        x1 = XYWH[:, 0] - XYWH[:, 2] * 0.5 + 0.5
        y1 = XYWH[:, 1] - XYWH[:, 3] * 0.5 + 0.5
        x2 = XYWH[:, 0] + XYWH[:, 2] * 0.5 - 0.5
        y2 = XYWH[:, 1] + XYWH[:, 3] * 0.5 - 0.5


        bboxes = torch.stack([x1, y1, x2, y2], dim=1)
        bboxes = bboxes.reshape(b,1,4)
        b = min(b, cfg.TRAIN.MaxShowBatch)
        box_img = draw_rect(search[:b, ...], bboxes[:b, ...])
        box_img = vutils.make_grid(box_img, normalize=True, scale_each=True)


        return box_img



    def forward(self, data):
        """ only used in training  对应到特征图上每个anchor的信息： , , overlap（正样本和所有anchor的IOU）
        """

        #如果不使用gru,对于模板和搜索区域均只在单帧上提取信息
        if not cfg.GRU.USE_GRU:

            template = data['template'].cuda()
            search = data['search'].cuda()
            label_cls = data['label_cls'].cuda()                #cls（此anchor是正样本：1、负样本：0、忽略：-1
            label_loc = data['label_loc'].cuda()                #delta（正样本框相对于anchor的编码偏移量
            label_loc_weight = data['label_loc_weight'].cuda()  #正样本对应的那些anchor的权重，其他位置为0

            # get feature
            zf = self.backbone(template)
            xf = self.backbone(search)

        #如果使用gru,模板需要在前t帧中累积提取，搜索区域只在最后一帧中提取
        else:

            zfs = [None] * self.grus.seq_in_len  # 多帧模板图z的特征f
            for i in range(self.grus.seq_in_len):
                # 每个data[i]中包含的信息为 'template','search','label_cls','label_loc','label_loc_weight','t_bbox','s_bbox''neg'
                zfs[i] = self.backbone(data[i]["template"].cuda())

            zfs=torch.stack(zfs,dim=1)            #将输入变为[n,t,c,h,w]的形式
            zf =self.grus(zfs).squeeze()          #grus输出为[n,1,c,h，w]的形式转化为【n,c,h,w】的形式

            #搜索区域只需要取模板序列组输入完成后的下一帧搜索区域图像就可以
            xf =  self.backbone(data[self.grus.seq_in_len]["search"].cuda())

#-------------------------------特征提取并行化-----------------------------------------------------

            # batch, _, _, _ = data[0]["template"].shape
            # zfs = [None] * (self.grus.seq_in_len)  # 多帧模板图z的特征f
            # for i in range(self.grus.seq_in_len):
            #     # 每个data[i]中包含的信息为 'template','search','label_cls','label_loc','label_loc_weight','t_bbox','s_bbox''neg'
            #     zfs[i] = data[i]["template"]
            #
            # #连续ｔ个序列在ｂａtch层面上并行，加快计算速度
            # zfs =  self.backbone( torch.cat(zfs,dim=0).cuda())
            # zfs =zfs.reshape(self.grus.seq_in_len, batch,  self.grus.input_channels, self.grus.input_height, self.grus.input_width)
            # zfs =zfs.permute(1, 0, 2, 3, 4).contiguous()
            #
            #
            # zf =self.grus(zfs).squeeze()          #grus输出为[n,1,c,h，w]的形式转化为【n,c,h,w】的形式
            #
            # #搜索区域只需要取模板序列组输入完成后的下一帧搜索区域图像就可以
            # xf =  self.backbone(data[self.grus.seq_in_len]["search"].cuda())

# ------------------------------------------------------------------------------------

            # 标签信息的提取方式和搜索区域的提取保持同步
            label_cls = data[self.grus.seq_in_len]['label_cls'].cuda()                #cls（此anchor是正样本：1、负样本：0、忽略：-1
            label_loc = data[self.grus.seq_in_len]['label_loc'].cuda()                #delta（正样本框相对于anchor的编码偏移量
            label_loc_weight = data[self.grus.seq_in_len]['label_loc_weight'].cuda()  #正样本对应的那些anchor的权重，其他位置为0



        if cfg.MASK.MASK:               #siamese mask
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:           #siamese rpn++
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)        #rpn相关计算

        # get loss
        cls_log = self.log_softmax(cls)             #softmax之后在log,将【0,1】之间的概率拉到【-inf,0】之间，后面紧接着的应该使用nlloss,  其中softmax+log+nllloss 等价于CrossEntropyLoss,这里之所以要拆解开的原因是我们需要按照anchor的mask来计算损失
        cls_loss = select_cross_entropy_loss(cls_log, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)



        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        # 是否计算GRU预测特征的损失
        if cfg.GRU.FeatLoss:
            zf_gt = self.backbone(data[self.grus.seq_in_len]["template"].cuda())
            feat_loss=weight_feat_loss(zf, zf_gt, data[self.grus.seq_in_len]["t_bbox"])
            outputs['total_loss'] += cfg.TRAIN.FEAT_WEIGHT * feat_loss
            outputs['feat_loss']    =feat_loss

            #传出去tensorboard监视看
            outputs['zf_gt'] = zf_gt
            outputs['zf'] = zf
            outputs['zfs'] = zfs




        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss


        if data[0]['iter']%cfg.TRAIN.ShowPeriod==0:
            #截断梯度,只取数据
            locd = loc.detach()
            clsd = cls.detach()
            outputs['box_img'] =self.show_result(clsd, locd, data[self.grus.seq_in_len]["search"])

        return outputs
