# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

import numpy as np

from pysot.utils.bbox import corner2center, center2corner


class Anchors:
    """
    This class generate anchors.
    """
    def __init__(self, stride, ratios, scales, image_center=0, size=0):
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.image_center = 0
        self.size = 0

        self.anchor_num = len(self.scales) * len(self.ratios)

        self.anchors = None

        self.generate_anchors()

    def generate_anchors(self):
        """
        generate anchors based on predefined configuration
        按照stride*stride作为参考基准，按照不同比例ratios,生产多个尺度的anchor
        例如：alexnet: stride =8, ratios = [0.33, 0.5, 1, 2, 3], 尺度为【8】
        输出的anchor的值为[-w*0.5, -h*0.5, w*0.5, h*0.5]，相当于是（x1,y1,x2,y2)坐标
        """
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride * self.stride
        count = 0
        for r in self.ratios:
            ws = int(math.sqrt(size*1. / r))
            hs = int(ws * r)

            for s in self.scales:
                w = ws * s
                h = hs * s
                self.anchors[count][:] = [-w*0.5, -h*0.5, w*0.5, h*0.5][:]
                count += 1

    def generate_all_anchors(self, im_c, size):
        """
        依据输入图像大小和rpn特征图大小size，以及generate_anchors生成的单点的anchor信息，为rpn输出层特征每一点生成anchor的相关信息
        im_c: image center （搜索区域图像的中心 255//2）
        size: image size  (输出特征图的大小17*17)
        """
        if self.image_center == im_c and self.size == size:
            return False
        self.image_center = im_c
        self.size = size

        a0x = im_c - size // 2 * self.stride                #在输入分辨下，模板与搜索区域中心对其，模板左上角的坐标
        ori = np.array([a0x] * 4, dtype=np.float32)
        zero_anchors = self.anchors + ori                   #为坐上角那个点产生anchor  大小为[n,4]

        x1 = zero_anchors[:, 0]         #大小为n
        y1 = zero_anchors[:, 1]
        x2 = zero_anchors[:, 2]
        y2 = zero_anchors[:, 3]
        #reshape为[n,1,1],中间这个1代表尺度，这里只选择了一个尺度
        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1),
                             [x1, y1, x2, y2])
        cx, cy, w, h = corner2center([x1, y1, x2, y2])   #shape 为【anchor_nums,1,1]
        #生成相对与左上角的点的偏移量
        disp_x = np.arange(0, size).reshape(1, 1, -1) * self.stride  #shape为【1,1，size】
        disp_y = np.arange(0, size).reshape(1, -1, 1) * self.stride

        cx = cx + disp_x    # shape为【anchor_nums,1,size]
        cy = cy + disp_y    # shape为 [anchor_nums,size,1]

        # broadcast 为每个点产生anchor
        zero = np.zeros((self.anchor_num, size, size), dtype=np.float32)
        cx, cy, w, h = map(lambda x: x + zero, [cx, cy, w, h])
        x1, y1, x2, y2 = center2corner([cx, cy, w, h])
        #生产两种类型的anchor,第一种是左上右下坐标类型的，第二种是中心点类型的，shape均为【4，anchor_num,size,size]
        self.all_anchors = (np.stack([x1, y1, x2, y2]).astype(np.float32),
                            np.stack([cx, cy, w,  h]).astype(np.float32))
        return True
