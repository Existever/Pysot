# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from pysot.core.config import cfg
from pysot.utils.bbox import IoU, corner2center
from pysot.utils.anchor import Anchors


class AnchorTarget:
    def __init__(self,):
        #按照给定的比例因子生成一个位置的多种anchor,输出shape为【anchor_num,4】其中输出的anchor的值为[-w * 0.5, -h * 0.5, w * 0.5, h * 0.5]
        self.anchors = Anchors(cfg.ANCHOR.STRIDE,       #8
                               cfg.ANCHOR.RATIOS,       # [0.33, 0.5, 1, 2, 3]
                               cfg.ANCHOR.SCALES)
        # 生成两种类型的anchor,第一种是左上右下坐标类型的，第二种是中心点类型的，shape均为【4，anchor_num,size,size]
        self.anchors.generate_all_anchors(im_c=cfg.TRAIN.SEARCH_SIZE//2,
                                          size=cfg.TRAIN.OUTPUT_SIZE)

    def __call__(self, target, size, neg=False):
        '''
        :param target:搜索区域坐标系下对应目标区域的bbox
        :param size:输出相关面特征图的大小
        :param neg:本次是否进行的是负样本对的训练
        :return: 对应到特征图上每个anchor的信息：cls（此anchor是正样本：1、负样本：0、忽略：-1）, delta（正样本框相对于anchor的编码偏移量）, delta_weight（正样本对应的那些anchor的权重，其他位置为0）, overlap（正样本和所有anchor的IOU）
        '''
        anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)

        # -1 ignore 0 negative 1 positive   （anchor标签信息，-1表示忽略，0表示负样本，1表示正样本）
        cls = -1 * np.ones((anchor_num, size, size), dtype=np.int64)
        delta = np.zeros((4, anchor_num, size, size), dtype=np.float32)
        delta_weight = np.zeros((anchor_num, size, size), dtype=np.float32)

        def select(position, keep_num=16):   #根据满足条件的位置模板，保留keep_num个有效的anchor（只做随机的选择，有点太简单了，至少按iou的高低来选择）
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        tcx, tcy, tw, th = corner2center(target)

        if neg:    #如果本轮训练的是负样本对，则认为在特征图上目标中心附近7*7的区域内的anchor都是负样本
            # l = size // 2 - 3
            # r = size // 2 + 3 + 1
            # cls[:, l:r, l:r] = 0

            cx = size // 2
            cy = size // 2
            cx += int(np.ceil((tcx - cfg.TRAIN.SEARCH_SIZE // 2) /
                      cfg.ANCHOR.STRIDE + 0.5))
            cy += int(np.ceil((tcy - cfg.TRAIN.SEARCH_SIZE // 2) /
                      cfg.ANCHOR.STRIDE + 0.5))
            l = max(0, cx - 3)       #目标中心附近7*7的区域
            r = min(size, cx + 4)
            u = max(0, cy - 3)
            d = min(size, cy + 4)
            cls[:, u:d, l:r] = 0

            neg, neg_num = select(np.where(cls == 0), cfg.TRAIN.NEG_NUM)
            cls[:] = -1
            cls[neg] = 0

            overlap = np.zeros((anchor_num, size, size), dtype=np.float32)
            return cls, delta, delta_weight, overlap

        anchor_box = self.anchors.all_anchors[0]         #all_anchors中已经提前为特征图的每个位置生成多种anchor的信息了。shape为【anchor_num,size,size】
        anchor_center = self.anchors.all_anchors[1]
        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], \
            anchor_box[2], anchor_box[3]
        cx, cy, w, h = anchor_center[0], anchor_center[1], \
            anchor_center[2], anchor_center[3]

        delta[0] = (tcx - cx) / w               #tcx,tcy表示的是目标的相关信息，而tx,ty表示的是anchor的相关信息，这里将位置信息编码为相对与anchor的偏移量或者比例使得网络更容易预测
        delta[1] = (tcy - cy) / h
        delta[2] = np.log(tw / w)
        delta[3] = np.log(th / h)

        overlap = IoU([x1, y1, x2, y2], target)  #计算这一个target和所有anchor的iou,

        pos = np.where(overlap > cfg.TRAIN.THR_HIGH)        #如果iou大于设定阈值（0.6），则认为是正样本
        neg = np.where(overlap < cfg.TRAIN.THR_LOW)        #如果iou小于设定阈值（0.3），则认为是负样本

        pos, pos_num = select(pos, cfg.TRAIN.POS_NUM)
        neg, neg_num = select(neg, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)

        cls[pos] = 1                                    #正样本位置设置为1,
        delta_weight[pos] = 1. / (pos_num + 1e-6)       #对正样本位置加权，方便计算loss时候对正样本位置取平均，其他位置 忽略

        cls[neg] = 0                                    #负样本设置为0，其他位置为初始值-1
        return cls, delta, delta_weight, overlap
