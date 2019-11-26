# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.backbone.alexnet import alexnetlegacy, alexnet
from pysot.models.backbone.mobile_v2 import mobilenetv2
from pysot.models.backbone.resnet_atrous import resnet18, resnet34, resnet50

BACKBONES = {
              'alexnetlegacy': alexnetlegacy,
              'mobilenetv2': mobilenetv2,
              'resnet18': resnet18,
              'resnet34': resnet34,
              'resnet50': resnet50,
              'alexnet': alexnet,
            }


def get_backbone(name, **kwargs):
    '''
    :param name: backbone的名字，也就是对应着在pysot.models.backbone文件下模块中的函数
    :param kwargs: 将一个不定长度的键值对作为参数传递给一个函数
    :return:返回BACKBONES中的某一种骨干网络，
    '''
    return BACKBONES[name](**kwargs)


