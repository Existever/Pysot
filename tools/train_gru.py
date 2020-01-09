
# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

from pysot.utils.lr_scheduler import build_lr_scheduler
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.distributed import dist_init, DistModule, reduce_gradients,\
        average_reduce, get_rank, get_world_size
from pysot.utils.model_load import load_pretrain, restore_from
from pysot.utils.average_meter import AverageMeter
from pysot.utils.misc import describe, commit
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.dataset import SeqTrkDataset
from pysot.core.config import cfg
from pysot.show import draw_rect




logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--cfg', type=str, default='config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_data_loader():
    '''
    :return: 建立train_loader,参数在config中指定
    '''

    logger.info("build train dataset")
    # train_dataset
    ##为feature map生成anchor的位置信息，通过json文件加载训练数据集（数据集合一视频为单位，保证每个视频至少一个跟踪目标，每个目标跟踪标注信息至少有一帧），设置数据增强的参数
    train_dataset = SeqTrkDataset(seq_input_len=cfg.GRU.SEQ_IN,seq_output_len=cfg.GRU.SEQ_OUT)
    logger.info("build SeqTrkDataset done")

    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              pin_memory=True,
                              sampler=train_sampler)
    return train_loader


def build_opt_lr(model, current_epoch=0):
    '''
    :param model:
    :param current_epoch:
    :return:
    '''
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
    else:
        for param in model.backbone.parameters():
            param.requires_grad = False
        for m in model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    #backbone的后几层的学习率要比rpn层的学习率小10倍，也就是在基础学习率BASE_LR上乘以LAYERS_LR
    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]
    #剩下的所有的和rpn相关的学习率都按照基础学习率作为基准
    if cfg.ADJUST.ADJUST:
        trainable_params += [{'params': model.neck.parameters(),
                              'lr': cfg.TRAIN.BASE_LR*cfg.GRU.NONE_GRU_LR_COFF}]

    trainable_params += [{'params': model.rpn_head.parameters(),
                          'lr': cfg.TRAIN.BASE_LR*cfg.GRU.NONE_GRU_LR_COFF}]

    if cfg.MASK.MASK:
        trainable_params += [{'params': model.mask_head.parameters(),
                              'lr': cfg.TRAIN.BASE_LR*cfg.GRU.NONE_GRU_LR_COFF}]

    if cfg.REFINE.REFINE:
        trainable_params += [{'params': model.refine_head.parameters(),
                              'lr': cfg.TRAIN.LR.BASE_LR*cfg.GRU.NONE_GRU_LR_COFF}]

    # 如果使用gru
    if cfg.GRU.USE_GRU:
        trainable_params += [{'params': model.grus.parameters(),
                              'lr': cfg.TRAIN.BASE_LR*cfg.GRU.LR_COFF }]



    #优化器使用带动量的SGD
    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    # optimizer = torch.optim.Adam(trainable_params,
    #                             weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler


def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, rpn_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            rpn_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/'+k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/'+k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/'+k.replace('.', '/'),
                             w_norm/(1e-20 + _norm), tb_index)
    tot_norm = feature_norm + rpn_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    rpn_norm = rpn_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/rpn', rpn_norm, tb_index)




def show_tensor(batch_data, global_iter,  tb_writer,outputs):
    '''
    :param batch_data: 输入的网络的数据
    :param global_iter:   tensorboard监视计数
    :param tb_writer:  tensorboard 的summarywriter
    :return:
    '''

    rank = get_rank()
    # global_iter = 0  # tensorboard监视计数
    # if rank==0:
    #     dataiter = iter(train_loader)
    #     data = next(dataiter)               #利用迭代器只取一个数据，用于构建图
    #     # tb_writer.add_graph(model,data)

    max_batch=cfg.TRAIN.MaxShowBatch            #tensorboard最多显示４个ｂａｔｃh
    batch, _, _, _ = batch_data[0]["template"].shape
    batch = min(batch, max_batch)

    if rank == 0 and global_iter%cfg.TRAIN.ShowPeriod==0:
        for i in range(cfg.GRU.SEQ_IN):
            xi =batch_data[i] # 每个data[i]中包含的信息为 'template','search','label_cls','label_loc','label_loc_weight','bbox','neg'

            tensor_t = draw_rect(xi["template"][0:batch], xi["t_bbox"][0:batch].view(batch,-1,4))
            tensor_s = draw_rect(xi["search"][0:batch], xi["s_bbox"][0:batch].view(batch,-1,4))
            tb_xi_template = vutils.make_grid(tensor_t, normalize=True,  scale_each=True)  # b c h w的图展开为多个图
            tb_writer.add_image('input/{}th_input_template'.format(i), tb_xi_template, global_iter)  # t_bbox是相对于模板坐标系的
            tb_xi_search = vutils.make_grid(tensor_s, normalize=True, scale_each=True)  # b c h w的图展开为多个图
            tb_writer.add_image('input/{}th_input_search'.format(i), tb_xi_search, global_iter)  # s_bbox是相对于搜索区域坐标系的


        for i in range(cfg.GRU.SEQ_OUT):
            xi = batch_data[i+cfg.GRU.SEQ_IN]  # 每个data[i]中包含的信息为 'template','search','label_cls','label_loc','label_loc_weight','bbox','neg'

            tensor_t = draw_rect(xi["template"][0:batch], xi["t_bbox"][0:batch].view(batch,-1,4))
            tensor_s = draw_rect(xi["search"][0:batch], xi["s_bbox"][0:batch].view(batch,-1,4))
            tb_xi_template = vutils.make_grid(tensor_t, normalize=True,  scale_each=True)  # b c h w的图展开为多个图
            tb_writer.add_image('input/{}th_output_template'.format(i+cfg.GRU.SEQ_IN), tb_xi_template, global_iter)  # t_bbox是相对于模板坐标系的
            tb_xi_search = vutils.make_grid(tensor_s, normalize=True, scale_each=True)  # b c h w的图展开为多个图
            tb_writer.add_image('input/{}th_output_search'.format(i+cfg.GRU.SEQ_IN), tb_xi_search, global_iter)  # s_bbox是相对于搜索区域坐标系的



        if  outputs['zf'] is not None:
            fb,fc,fh,fw= outputs['zf'].shape
            fc =(min(fc,9)//3)*3            #最多显示9个通道的数据
            for i in range(0,fc,3):
                tb_feat = vutils.make_grid(outputs['zf'][0:batch,i:i+3,...], normalize=True, scale_each=True)  # b c h w的图展开为多个图
                tb_writer.add_image('feature/{}th_feat'.format(i), tb_feat, global_iter)                # t_bbox是相对于模板坐标系的

        if outputs['zfs'] is not None:
            _, ft, _, _, _ = outputs['zfs'].shape
            for t in range(ft):
                feat = outputs['zfs'][:, t, ...]
                fb, fc, fh, fw = feat.shape
                fc = (min(fc, 9) // 3) * 3  # 最多显示9个通道的数据
                for i in range(0, fc, 3):
                    tb_feat = vutils.make_grid(feat[0:batch, i:i + 3, ...], normalize=True,
                                               scale_each=True)  # b c h w的图展开为多个图
                    tb_writer.add_image('feature/{}th_{}feat'.format(i, t), tb_feat, global_iter)  # t_bbox是相对于模板坐标系的

        if outputs['zf_gt'] is not None:
            fb, fc, fh, fw = outputs['zf_gt'].shape
            fc = (min(fc, 9) // 3) * 3  # 最多显示9个通道的数据
            for i in range(0, fc, 3):
                tb_feat_gt = vutils.make_grid(outputs['zf_gt'][0:batch,i:i+3,...], normalize=True, scale_each=True)  # b c h w的图展开为多个图
                tb_writer.add_image('feature/{}th_feat_gt'.format(i), tb_feat_gt, global_iter)  # t_bbox是相对于模板坐标系的

        if  outputs['box_img'] is not None:
            tb_writer.add_image('predict/box_img', outputs['box_img'], global_iter)  # t_bbox是相对于模板坐标系的


def train(train_loader, model, optimizer, lr_scheduler, tb_writer):
    '''
    :param train_loader:
    :param model:
    :param optimizer:
    :param lr_scheduler:
    :param tb_writer:
    :return:
    '''
    cur_lr = lr_scheduler.get_cur_lr()        #获得当前学习率
    rank = get_rank()

    average_meter = AverageMeter()

    def is_valid_number(x):
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = get_world_size()
    num_per_epoch = len(train_loader.dataset) // cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and get_rank() == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    logger.info("model\n{}".format(describe(model.module)))   #打印模型
    end = time.time()
    for idx, data in enumerate(train_loader):

        if epoch != idx // num_per_epoch + start_epoch:       #每个epoch的跳变沿进行一次模型存储
            epoch = idx // num_per_epoch + start_epoch

            if get_rank() == 0:
                torch.save(
                        {'epoch': epoch,
                         'state_dict': model.module.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        cfg.TRAIN.SNAPSHOT_DIR+'/checkpoint_e%d.pth' % (epoch))

            if epoch == cfg.TRAIN.EPOCH:
                return

            # 如果达到第10个epoch后，则要开始微调backbone的后面3层，要重新设置一下哪些参数是可训练的，哪些参数是不动的，学习率的调整因子
            if cfg.BACKBONE.TRAIN_EPOCH == epoch:
                logger.info('start training backbone.')
                optimizer, lr_scheduler = build_opt_lr(model.module, epoch)
                logger.info("model\n{}".format(describe(model.module)))

            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()
            logger.info('epoch: {}'.format(epoch+1))

        tb_idx = idx            #tensor board的idx
        if idx % num_per_epoch == 0 and idx != 0:
            for idx, pg in enumerate(optimizer.param_groups):       #将优化器中的学习率添加到tensorboard中监视
                logger.info('epoch {} lr {}'.format(epoch+1, pg['lr']))
                if rank == 0:
                    tb_writer.add_scalar('lr/group{}'.format(idx+1),   pg['lr'], tb_idx)

        data_time = average_reduce(time.time() - end)
        if rank == 0:
            tb_writer.add_scalar('time/data', data_time, tb_idx)

       # show_tensor(data, tb_idx, tb_writer)  # 只看输入数据，在tensorboard中显示输入数据
        data[0]['iter']=tb_idx                           #添加监视用
        outputs = model(data)
        loss = outputs['feat_loss']
        # loss = outputs['total_loss']
        show_tensor(data, tb_idx, tb_writer,outputs)  #输入输出都看，在tensorboard中显示输入数据


        if is_valid_number(loss.data.item()):           #判断损失是否是合法数据，滤掉nan,+inf，>10000的这样的损失
            optimizer.zero_grad()
            loss.backward()
            reduce_gradients(model)                     #分发梯度

            if rank == 0 and cfg.TRAIN.LOG_GRADS:       #对梯度信息监视
                log_grads(model.module, tb_writer, tb_idx)

            # clip gradient
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            optimizer.step()

        batch_time = time.time() - end
        batch_info = {}
        batch_info['batch_time'] = average_reduce(batch_time)
        batch_info['data_time'] = average_reduce(data_time)
        for k, v in sorted(outputs.items()):
            if k is 'zf' or k is  'zf_gt'or k is  'zfs'or k is  'box_img':
                pass
            else:
             batch_info[k] = average_reduce(v.data.item())

        average_meter.update(**batch_info)

        if rank == 0:
            for k, v in batch_info.items():
                tb_writer.add_scalar(k, v, tb_idx)

            if (idx+1) % cfg.TRAIN.PRINT_FREQ == 0:
                info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(
                            epoch+1, (idx+1) % num_per_epoch,
                            num_per_epoch, cur_lr)
                for cc, (k, v) in enumerate(batch_info.items()):
                    if cc % 2 == 0:
                        info += ("\t{:s}\t").format(
                                getattr(average_meter, k))
                    else:
                        info += ("{:s}\n").format(
                                getattr(average_meter, k))
                logger.info(info)
                print_speed(idx+1+start_epoch*num_per_epoch, average_meter.batch_time.avg,cfg.TRAIN.EPOCH * num_per_epoch)
        end = time.time()


def main():
    rank, world_size = dist_init()
    logger.info("init done")


    # load cfg
    cfg.merge_from_file(args.cfg)                   #将core下面的config配置文件与experiments里面的配置文件融合，因实验不同，修改默认参数
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):   #在tools 路径下建立log日志文件家
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)            #添加到log日志里面
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',              #在logs文件夹下建立log.txt文本记录控制台信息
                             os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                             logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    # create model
    model = ModelBuilder().cuda().train()
    dist_model = DistModule(model)

    # load pretrained backbone weights
    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, '../', cfg.BACKBONE.PRETRAINED)
        load_pretrain(model.backbone, backbone_path)

    # create tensorboard writer
    if rank == 0 and cfg.TRAIN.LOG_DIR:
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        tb_writer = None

    # build dataset loader
    train_loader = build_data_loader()

    # build optimizer and lr_scheduler,使用SGD优化器（适合探索新的结构），以START_EPOCH为基准设置学习率，方便中断后再次训练
    optimizer, lr_scheduler = build_opt_lr(dist_model.module,
                                           cfg.TRAIN.START_EPOCH)

    # resume training
    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, cfg.TRAIN.RESUME)
    # load pretrain
    elif cfg.TRAIN.PRETRAINED:
        load_pretrain(model, cfg.TRAIN.PRETRAINED)
    dist_model = DistModule(model)

    logger.info(lr_scheduler)
    logger.info("model prepare done")

    # start training
    train(train_loader, dist_model, optimizer, lr_scheduler, tb_writer)


if __name__ == '__main__':
    seed_torch(args.seed)
    main()
