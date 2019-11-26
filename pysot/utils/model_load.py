# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import torch


logger = logging.getLogger('global')


def check_keys(model, pretrained_state_dict):
    '''
    :param model: 搭建的网络模型
    :param pretrained_state_dict: 从预训练中的模型加载的模型参数
    :return: （检查当前加载的网络模型参数和预训练的模型是否一致）
    '''
    ckpt_keys = set(pretrained_state_dict.keys())       #将预训练的网络各层变量名放在集合中
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys       #求交集
    unused_pretrained_keys = ckpt_keys - model_keys     #求差集得到预训练的模型中没有使用的层
    missing_keys = model_keys - ckpt_keys               #求缺失的网络层参数
    # filter 'num_batches_tracked'
    missing_keys = [x for x in missing_keys
                    if not x.endswith('num_batches_tracked')]    #从PyTorch 0.4.1开始, BN层中新增加了一个参数 track_running_stats，训练时用来统计训练时的forward过的min-batch数目,每经过一个min-batch, track_running_stats+=1
    if len(missing_keys) > 0:
        logger.info('[Warning] missing keys: {}'.format(missing_keys))
        logger.info('missing keys:{}'.format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        logger.info('[Warning] unused_pretrained_keys: {}'.format(
            unused_pretrained_keys))
        logger.info('unused checkpoint keys:{}'.format(
            len(unused_pretrained_keys)))
    logger.info('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, \
        'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters
    share common prefix 'module.' '''
    logger.info('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain(model, pretrained_path):
    logger.info('load pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))   #将模型文件加载到所有可见的文件中去
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'],       #去掉老版本pytorch中的module.这个前缀
                                        'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    try:
        check_keys(model, pretrained_dict)
    except:
        logger.info('[Warning]: using pretrain as features.\
                Adding "features." as prefix')
        new_dict = {}
        for k, v in pretrained_dict.items():
            k = 'features.' + k
            new_dict[k] = v
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def restore_from(model, optimizer, ckpt_path):
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path,
        map_location=lambda storage, loc: storage.cuda(device))
    epoch = ckpt['epoch']

    ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
    check_keys(model, ckpt_model_dict)
    model.load_state_dict(ckpt_model_dict, strict=False)

    check_keys(optimizer, ckpt['optimizer'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, epoch
