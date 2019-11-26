# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.head.mask import MaskCorr, Refine
from pysot.models.head.rpn import UPChannelRPN, DepthwiseRPN, MultiRPN

RPNS = {
        'UPChannelRPN': UPChannelRPN,           #siamese rpn里面使用，先将通道数目按照anchor成倍提升，再分组卷积得到与anchor种类数相对应的输出通道数目
        'DepthwiseRPN': DepthwiseRPN,
        'MultiRPN': MultiRPN
       }

MASKS = {
         'MaskCorr': MaskCorr,
        }

REFINE = {
          'Refine': Refine,
         }


def get_rpn_head(name, **kwargs):
    return RPNS[name](**kwargs)


def get_mask_head(name, **kwargs):
    return MASKS[name](**kwargs)


def get_refine_head(name):
    return REFINE[name]()
