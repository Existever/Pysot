# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from pysot.utils.bbox import center2corner, Center
from pysot.datasets.anchor_target import AnchorTarget
from pysot.datasets.augmentation import Augmentation
from pysot.core.config import cfg

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)

# 子数据集
class SubDataset(object):
    def __init__(self, name, root, anno, frame_range, num_use, start_idx):
        '''
        这是一个生成子数据集的类:
        标准化bbox为(x,y,w,h),删除空目标,删除空video
        生成标签label
        原coco2014训练集量为82783,清洗后数据集量为82081
        :param name: 数据集名称  COCO
        :param root: crop后的数据集路径
        :param anno: crop后生成的标注信息
        :param frame_range:
        :param num_use: 控制使用数据集的数量   若num_use=-1  数量等于有效数据集全部  否则等于num_use
        :param start_idx: 开始的帧号
        '''
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.root = os.path.join(cur_path, '../../', root)  # 裁剪后的图的路径
        self.anno = os.path.join(cur_path, '../../', anno)  # 新生成的json路径
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        logger.info("loading " + name)
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)
            meta_data = self._filter_zero(meta_data)  # 将jeson的数据中bbox改为(x,y,w,h)格式
        '''
        对应关系:
              video   对应每一张图像 例如train2014/COCO_train2014_000000000009
              track   对应每一张图相中的目标  例如 00:第00个目标  01:第01个目标
        '''
        for video in list(meta_data.keys()):   # 判断是不是每个序列的目标都有标注信息000000,若没有标注信息,删除这个目标
            for track in meta_data[video]:
                frames = meta_data[video][track]
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))   # filter为过滤函数,判断gt是不是均为数字,若是返回1 否则返回0
                frames.sort()
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):  # 判断是不是每个序列(子文件夹)都有目标,若没有目标,就删除这个序列
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data    # 所有目标的gt
        self.num = len(self.labels)    # video的数量(图像的数量)
        self.num_use = self.num if self.num_use == -1 else self.num_use  # 等于可用video数量
        self.videos = list(meta_data.keys())  # 图像序列号
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'
        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
        '''
        对json数据做处理,转换bbox为(x,y,w,h)模式
        :param meta_data: 原json
        :return: 新json
        '''
        meta_data_new = {}
        # 拆第一级目录
        for video, tracks in meta_data.items():   # video为单一图像文件夹  tracks是字典,key为一张图像中目标索引,value为目标的gt
            new_tracks = {}
            # 拆二级目录
            for trk, frames in tracks.items():     # trk为目标索引00 01 02...  frames为目标gt
                new_frames = {}
                # 拆三级目录
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    # 添加新三级目录
                    new_frames[frm] = bbox
                # 添加新二级目录
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            # 添加新一级目录
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        '''
        打印载入数据集的日志
        '''
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        '''
        创建数据集中目标的索引
        :return: 乱序的索引序列
        '''
        lists = list(range(self.start_idx, self.start_idx + self.num))  # 对应数据集数据个数的list
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info['frames']
        template_frame = np.random.randint(0, len(frames))
        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames)-1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        search_frame = np.random.choice(search_range)
        return self.get_image_anno(video_name, track, template_frame), \
            self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class TrkDataset(Dataset):
    '''
    生成所有数据集list:  all_dataset
    数据增强
    按照epoch设置数据集的量: ALexnet需要epoch为50 只使用了coco,有80000个video,数据集有4000000个video
    并洗牌
    '''
    def __init__(self,):
        super(TrkDataset, self).__init__()

        desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        if desired_size != cfg.TRAIN.OUTPUT_SIZE:
            raise Exception('size not match!')

        # create anchor target
        self.anchor_target = AnchorTarget()

        # create sub dataset
        '''
        创建数据集all_dataset列表
        用SubDataset类去实例化子数据集对象
        并放入all_dataset列表中
        '''
        self.all_dataset = []
        start = 0                # 开始的帧号
        self.num = 0             # target的数量
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)  # 载入数据集设置
            sub_dataset = SubDataset(                 # 实例化对象
                    name,
                    subdata_cfg.ROOT,
                    subdata_cfg.ANNO,
                    subdata_cfg.FRAME_RANGE,
                    subdata_cfg.NUM_USE,   # 使用多少个数据集,若为-1,有效的均使用
                    start
                )
            start += sub_dataset.num     # 记录结束点,也是下一个数据集的开始记录点
            self.num += sub_dataset.num_use   # 记录目标数据量,每个数据集可用的数据量的累加
            sub_dataset.log()
            self.all_dataset.append(sub_dataset)
            # 如果第一个为COCO 就认为只使用了一个数据集
            if name == 'COCO':
                break

        # data augmentation
        self.template_aug = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT,
                cfg.DATASET.TEMPLATE.SCALE,
                cfg.DATASET.TEMPLATE.BLUR,
                cfg.DATASET.TEMPLATE.FLIP,
                cfg.DATASET.TEMPLATE.COLOR
            )
        self.search_aug = Augmentation(
                cfg.DATASET.SEARCH.SHIFT,
                cfg.DATASET.SEARCH.SCALE,
                cfg.DATASET.SEARCH.BLUR,
                cfg.DATASET.SEARCH.FLIP,
                cfg.DATASET.SEARCH.COLOR
            )
        # videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH       # 600000个videos为一个epoch
        videos_per_epoch = -1            # 只使用了一个数据集 coco的videos有80000个
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH      # 训50轮 一轮即遍历所有数据
        self.pick = self.shuffle()

    def shuffle(self):
        '''
        四个数据集的vedio各取一次为一个epoch
        取整数轮次,当所有的数据(vedio)大于要求数据量时,返回取出数据的索引
        :return:
        '''
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:  # 取出一个子数据集
                sub_p = sub_dataset.pick
                p += sub_p                        # 四个数据集的video累加
            np.random.shuffle(p)                  # 四个数据集的video洗牌
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        dataset, index = self._find_dataset(index)

        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        # get one dataset
        if neg:
            template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)

        # get image
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])

        # get bounding box
        template_box = self._get_bbox(template_image, template[1])
        search_box = self._get_bbox(search_image, search[1])

        # augmentation
        template, _ = self.template_aug(template_image,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)

        search, bbox = self.search_aug(search_image,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)

        # get labels
        cls, delta, delta_weight, overlap = self.anchor_target(
                bbox, cfg.TRAIN.OUTPUT_SIZE, neg)
        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)
        return {
                'template': template,
                'search': search,
                'label_cls': cls,
                'label_loc': delta,
                'label_loc_weight': delta_weight,
                'bbox': np.array(bbox)
                }
