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


class SubDataset(object):
    '''

    '''
    def __init__(self, name, root, anno, frame_range, num_use, start_idx):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.root = os.path.join(cur_path, '../../', root)
        self.anno = os.path.join(cur_path, '../../', anno)
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        logger.info("loading " + name)   #json文件以视频id键,每个视频里面又分为多个跟踪目标，每个跟踪目标有多帧，每帧以帧号为键，标注信息为值，对于只有一张图片的coco数据集，一张图作为一个video
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)
            meta_data = self._filter_zero(meta_data)
        #再次做合法性质检查，对于帧号不是数字的滤出掉，对于每个跟踪目标而言，帧号按照从小到达排序，保证每个视频中是少有一个跟踪目标，一个跟踪目标至少有一帧标注信息
        for video in list(meta_data.keys()):
            for track in meta_data[video]:
                frames = meta_data[video][track]
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))   #filter滤波frames.keys（）是否为数字，再通过map转换为int，因为后面要排序
                frames.sort()                   #对于跟踪目标而言，帧号从小达到排序
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'
        self.pick = self.shuffle()
    #滤出掉bbox中w,h小于0的标注，保证一个视频中至少有一个跟踪目标，一个跟踪目标至少有一帧标注信息
    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():   #key和value分别表示，视频id，和视频中跟踪目标的id
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}                  #对于每一个跟踪目标的标注信息，滤出标注bbox中w,h<0的bbox
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:             #如果滤波之后某个跟踪目标，标注帧数>0，认为是一个有效的跟踪序列标注信息
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:                 #如果滤波之后某个视频跟踪目标的个数>0,则认为这个视频的标注是一个有效的信息
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        '''
        :param video:
        :param track:
        :param frame:
        :return:
        '''
        frame = "{:06d}".format(frame)
        image_path = os.path.join(self.root, video,         #'/root/video/000000.01.x.jpg'  图片的命名按照 [帧号][跟踪目标编号].x.jpg命名，其中x表示搜索区域
                                  self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video][track][frame]      #标签按照视频-->跟踪目标-->跟踪帧号三个维度定义
        return image_path, image_anno


    def get_positive_pair(self, index):
        '''
        #按照给定视频的序号，从该视频中随机选择一个跟踪目标，从这个跟踪目标中选择一帧作为跟踪模板帧，
        在跟踪模板帧附近随机选择一帧作为搜索区域帧，无论是模板图还是搜索区域图，都是读取一张完整的大图
        :param index:
        :return: 【template_info,search_info】,其中xxx_info=【img_path, bbox】两个信息
        '''
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
    #从指定的视频中随机选择一个跟踪目标，从跟踪目标中随机选择一帧作为跟踪模板
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

#为feature map生成anchor的位置信息，通过json文件加载训练数据集（数据集合一视频为单位，保证每个视频至少一个跟踪目标，每个目标跟踪标注信息至少有一帧），设置数据增强的参数
class TrkDataset(Dataset):
    '''

    '''
    def __init__(self,):
        super(TrkDataset, self).__init__()
        #特征图输出size为：（搜索区域尺寸 -- 模板尺寸)/stride +1  ,没有padding
        desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        if desired_size != cfg.TRAIN.OUTPUT_SIZE:
            raise Exception('size not match!')

        # create anchor target（初始化事为每个点设置anchor的形状，调用的时候确定哪些anchor是正样本，哪些anchor是负样本）
        self.anchor_target = AnchorTarget()

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                    name,
                    subdata_cfg.ROOT,
                    subdata_cfg.ANNO,
                    subdata_cfg.FRAME_RANGE,
                    subdata_cfg.NUM_USE,
                    start
                )
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        # data augmentation（数据增强初始化参数，对于模板的数据增强参数，和搜索区域的数据增强参数不一样，例如模板平移范围为4个像素，而搜索区域平移范围为64个像素）
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
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        '''
        :return:
        '''
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        '''
        :param index:
        :return:
        '''
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:         #这个地方不妥吧？
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        '''默认模板图像位于整个图像的中心，将gt标注的bbox加上0.5倍大小的上下文图像内容作为模板区域，认为是网络训练的模板区域，缩放到127*127，输出缩放后的相对与图像中心的模板坐标
        :param image:
        :param shape:
        :return:
        '''
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5            #上下文占用的比例，gt构成的box再加上一定比例的上下文图像内容，认为是模板区域
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z    #模板在网络中默认为127*127大小的，在crop数据集合的时候，把具有上下问的模板区域resize成了127*127，所以w,h要同比例缩放
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2         #因为在制作数据集合的时候，模板区域已经默认对齐到图像中心
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        dataset, index = self._find_dataset(index)

        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()           #通过随机产生的值决定是否将输入图像变换灰度
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()              #通过随机产生的只决定是使用正样本对和负样本对

        # get one dataset
        if neg:#负样本对：模板在当前数据中随机按照索引抽取一个视频，随机选择其中一个跟踪目标，从这个目标中随机选择一帧，作为模板，而搜索区域则先进行数据集的随机选，再按照前面的规则继续选择搜索区域
            template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)

        # get image  template【0】中是图片路径
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])

        # get bounding box template【1】中是bbox标签信息，这里生成的template_box和search_box本质上都是对应着模板区域的大小，
        # 因为输入的图片的数据已经将gt_box和模板的size(127*127)在gen_crp过程中已经对应了起来，如果模板不做数据增强的话，这一步
        # 直接从中心扣取127*127的区域就是模板（其实是包含0.5倍的上下文图像内容的），扣取255*255d的区域就是搜索区域，但是在后面的
        # 数据增强过程中对目标的尺度会有轻微的调整，调整之后依然要resize到127*127的过程，这个轻微的调整是需要网络能够通过学习去适应的，
        template_box = self._get_bbox(template_image, template[1])
        search_box = self._get_bbox(search_image, search[1])

        # augmentation（扣取模板信息和搜索区域的信息，返回的box信息是转化到扣取图像坐标系下的信息）
        template, _ = self.template_aug(template_image,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)

        search, bbox = self.search_aug(search_image,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)

        # get labels  对应到特征图上每个anchor的信息：cls（此anchor是正样本：1、负样本：0、忽略：-1）, delta（正样本框相对于anchor的编码偏移量）, delta_weight（正样本对应的那些anchor的权重，其他位置为0）, overlap（正样本和所有anchor的IOU）
        cls, delta, delta_weight, overlap = self.anchor_target(
                bbox, cfg.TRAIN.OUTPUT_SIZE, neg)
        template = template.transpose((2, 0, 1)).astype(np.float32)     #bgr -->rgb
        search = search.transpose((2, 0, 1)).astype(np.float32)
        return {
                'template': template,
                'search': search,
                'label_cls': cls,
                'label_loc': delta,
                'label_loc_weight': delta_weight,
                'bbox': np.array(bbox)
                }
