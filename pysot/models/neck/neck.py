# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn

#siamese rpn++中fig3中，将backbone的通道数通过1*1的卷积调整，调整到与anchor的个数相互对应
#同时模板通道输入的feature map大小为15*15，作为卷积核太大，这里只裁剪7*7的中心区域，当然感受野覆盖了整个图
class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )
        self.center_size = center_size

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:                   #输入的feature map只取7*7的中心
            l = (x.size(3) - self.center_size) // 2
            r = l + self.center_size
            x = x[:, :, l:r, l:r]
        return x

#siamese rpn++ 中最后输出的anchor包括多个调整分支，这里将多个调整封装在一块
class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels, center_size=7):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0],
                                          out_channels[0],
                                          center_size)
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),
                                AdjustLayer(in_channels[i],
                                            out_channels[i],
                                            center_size))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                out.append(adj_layer(features[i]))
            return out
