# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "siamrpn_r50_l234_dwxcorr"    #原始架构

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# Anchor Target
# Positive anchor threshold（anchor与gt之间的IOU大于该阈值,则认为该anchor为真实样本）
__C.TRAIN.THR_HIGH = 0.6

# Negative anchor threshold
__C.TRAIN.THR_LOW = 0.3

# Number of negative  如果某次训练进行负样本对的训练，则产生的负样本对的最大值
__C.TRAIN.NEG_NUM = 16

# Number of positive    通过anchor选择正样本的个数的最大值
__C.TRAIN.POS_NUM = 16

# Number of anchors per images   通过anchor选择正样本负样本总数的最大值
__C.TRAIN.TOTAL_NUM = 64


__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 255

__C.TRAIN.BASE_SIZE = 8

__C.TRAIN.OUTPUT_SIZE = 25  #输出相关面的尺寸

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = ''

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './snapshot'

__C.TRAIN.MaxShowBatch =4           #tensorboard最大的显示batch数目

__C.TRAIN.ShowPeriod =200            #每多少次迭代像tensorboard中添加一次图像及结果的显示

__C.TRAIN.EPOCH = 20

__C.TRAIN.START_EPOCH = 0       #开始的epoch用于从不同的断点恢复，调整学习率

__C.TRAIN.BATCH_SIZE = 4

__C.TRAIN.NUM_WORKERS = 16

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0      #位置损失的权重

__C.TRAIN.LOC_WEIGHT = 1.2      #类别算是的权重

__C.TRAIN.FEAT_WEIGHT =0.1      #GRU特征权重

__C.TRAIN.MASK_WEIGHT = 1

__C.TRAIN.PRINT_FREQ = 20       #控制台每多少次数打印一次信息

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005   #基础学习率

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'  #warmup之后使用log的方式调整学习率

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()    #warmup阶段使用step的方式调整学习率

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# Random shift see [SiamPRN++](https://arxiv.org/pdf/1812.11703)
# for detail discussion
__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18

#BLUR非0，且正太分布产生的随机数小于BLUR才进行模糊，配置为0表示不模糊
__C.DATASET.SEARCH.BLUR = 0.0

# FLIP非0，且正太分布产生的随机数小于flip才进行翻转，对于siampeserpn,如果模板翻转搜索区域也要翻转，
# 程序里没有保持一致，所以统一选择不翻转，配置为0表示不翻转
__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

# Sample Negative pair see [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
# for detail discussion
__C.DATASET.NEG = 0.2

# improve tracking performance for otb100
#（如果随机选择过程要进行灰度化，则先将彩色图像转化为灰度，在从灰度转化为3通道“彩图”）
__C.DATASET.GRAY = 0.0


__C.DATASET.NAMES = (['GOT10K','VID'])    # only use coco for training zsy
__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = '/media/rainzsy/00024268000F00F7/coco/crop511'
__C.DATASET.COCO.ANNO = '/media/rainzsy/00024268000F00F7/coco/train2017.json'
__C.DATASET.COCO.FRAME_RANGE = 1  #模板图帧号为k,则搜索区域图在视频序列，在[k-FRAME_RANGE,k+FRAME_RANGE】范围内随机选择
__C.DATASET.COCO.NUM_USE = -1     #使用多少个视频，-1表示使用全部，否则按照指定个数使用，如果指定个数大于总个数，在标签shuffle的时候会随机重复取，直到满足设定个数要求

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = 'training_dataset/vid/crop511'
__C.DATASET.VID.ANNO = 'training_dataset/vid/train_mini.json'
__C.DATASET.VID.FRAME_RANGE = 10                                    #搜索区域对应图像帧号，在模板图像帧号正负FRAME_RANGE内
__C.DATASET.VID.NUM_USE = 100000  # repeat until reach NUM_USE


__C.DATASET.GOT10K = CN()
__C.DATASET.GOT10K.ROOT = '/home/rainzsy/datasets/got10k/crop511'
__C.DATASET.GOT10K.ANNO = 'training_dataset/got10k/train.json'
__C.DATASET.GOT10K.FRAME_RANGE = 10                                    #搜索区域对应图像帧号，在模板图像帧号正负FRAME_RANGE内
__C.DATASET.GOT10K.NUM_USE = -1  # repeat until reach NUM_USE

# __C.DATASET.NAMES = ('VID', 'COCO', 'DET', 'YOUTUBEBB')
#
# __C.DATASET.VID = CN()
# __C.DATASET.VID.ROOT = 'training_dataset/vid/crop511'
# __C.DATASET.VID.ANNO = 'training_dataset/vid/train.json'
# __C.DATASET.VID.FRAME_RANGE = 100
# __C.DATASET.VID.NUM_USE = 100000  # repeat until reach NUM_USE
#
# __C.DATASET.YOUTUBEBB = CN()
# __C.DATASET.YOUTUBEBB.ROOT = 'training_dataset/yt_bb/crop511'
# __C.DATASET.YOUTUBEBB.ANNO = 'training_dataset/yt_bb/train.json'
# __C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
# __C.DATASET.YOUTUBEBB.NUM_USE = -1  # use all not repeat
#
# __C.DATASET.COCO = CN()
# __C.DATASET.COCO.ROOT = 'training_dataset/coco/crop511'
# __C.DATASET.COCO.ANNO = 'training_dataset/coco/train2017.json'
# __C.DATASET.COCO.FRAME_RANGE = 1
# __C.DATASET.COCO.NUM_USE = -1
#
# __C.DATASET.DET = CN()
# __C.DATASET.DET.ROOT = 'training_dataset/det/crop511'
# __C.DATASET.DET.ANNO = 'training_dataset/det/train.json'
# __C.DATASET.DET.FRAME_RANGE = 1
# __C.DATASET.DET.NUM_USE = -1

__C.DATASET.VIDEOS_PER_EPOCH = 600000    #每个epoch训练使用的视频个数，如果不够，则通过shuffle重复选取
# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'res50'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Train layers  当 current_epoch >= cfg.BACKBONE.TRAIN_EPOCH时这些层才会参与训练
__C.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1  #backbone开始训练时，他的学习率要比rpn层的学习率小，也就是在基础学习率BASE_LR上乘以LAYERS_LR

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10   #注意当 current_epoch >= cfg.BACKBONE.TRAIN_EPOCH时设定的backbone的后几层才会参与训练，否则梯度不反向传播，且BN层参数不要更新


# ------------------------------------------------------------------------ #
# GRU param  options
# ------------------------------------------------------------------------ #
__C.GRU = CN()
__C.GRU.USE_GRU = False       #是否使用GRU模块
__C.GRU.SEQ_IN = 3         #GRU连续输入序列长度
__C.GRU.SEQ_OUT = 1         #GRU输出预测序列长度,对于跟踪问题只设置为1
__C.GRU.FeatLoss = False    #GRU计算特征图是否计算损失
__C.GRU.LR_COFF = 0.1      #联合训练的时候GRU模块的学习率比例系数







# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"

# ------------------------------------------------------------------------ #
# RPN options
# ------------------------------------------------------------------------ #
__C.RPN = CN()

# RPN type
__C.RPN.TYPE = 'MultiRPN'

__C.RPN.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# mask options
# ------------------------------------------------------------------------ #
__C.MASK = CN()

# Whether to use mask generate segmentation
__C.MASK.MASK = False

# Mask type
__C.MASK.TYPE = "MaskCorr"

__C.MASK.KWARGS = CN(new_allowed=True)

__C.REFINE = CN()

# Mask refine
__C.REFINE.REFINE = False

# Refine type
__C.REFINE.TYPE = "Refine"

# ------------------------------------------------------------------------ #
# Anchor options
# ------------------------------------------------------------------------ #
__C.ANCHOR = CN()

# Anchor stride
__C.ANCHOR.STRIDE = 8

# Anchor ratios
__C.ANCHOR.RATIOS = [0.33, 0.5, 1, 2, 3]

# Anchor scales
__C.ANCHOR.SCALES = [8]

# Anchor number
__C.ANCHOR.ANCHOR_NUM = len(__C.ANCHOR.RATIOS) * len(__C.ANCHOR.SCALES)


# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamRPNTracker'

# Scale penalty   尺度惩罚因子，在测试时候，跟踪过程中使用
__C.TRACK.PENALTY_K = 0.04

# Window influence  hanning 窗口影响因子，在score 中加入0.44倍的hanning窗口，因子，在测试时候，跟踪过程中使用
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate 对于尺度加权融合的时候，  在测试时候，跟踪过程中使用
__C.TRACK.LR = 0.4

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255       #相当于是搜索区域面积

# Base size
__C.TRACK.BASE_SIZE = 8

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5     #搜索区域的占宽（高）的比例

# Long term lost search size
__C.TRACK.LOST_INSTANCE_SIZE = 831

# Long term confidence low
__C.TRACK.CONFIDENCE_LOW = 0.85

# Long term confidence high
__C.TRACK.CONFIDENCE_HIGH = 0.998

# Mask threshold
__C.TRACK.MASK_THERSHOLD = 0.30

# Mask output size
__C.TRACK.MASK_OUTPUT_SIZE = 127



