# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker


class SiamRPN_GRU_Tracker(SiameseTracker):
    def __init__(self, model):
        #初始化输入就是加载完成的参数模型
        super(SiamRPN_GRU_Tracker, self).__init__()

        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)     #生成和输出特征图大小相同的hanning窗
        window = np.outer(hanning, hanning)       #一维度的hanning窗口通过外积得到二位hanning 窗口
        self.window = np.tile(window.flatten(), self.anchor_num)        #为多个anchor生成hanning窗口
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()                         #模型做只做前向，不更新bn参数
        self.template_idx = 0                       #模板的索引号

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,cfg.ANCHOR.RATIOS, cfg.ANCHOR.SCALES)



        anchor = anchors.anchors                        #shape 为 【anchor_num,4】,输出的是【-0.5w,-0.5h,0.5w,0.5h],左上右下的坐标类型
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]

        # 得到w,h的信息，x,y为0,0，下面的过程是生成anchor的x,y信息，最左上角点坐标为【-0.5w,-0.5h],中心坐标为0,0
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)

        #下面为特征图上的每个格点生成中心点x,y的坐标
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride

        #使用meshgrid生成连续的格点坐标，xx,yy大小均为【size,size]
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        #为多种anchor生成连续的格点坐标
        xx = np.tile(xx.flatten(), (anchor_num, 1)).flatten()
        yy=np.tile(yy.flatten(), (anchor_num, 1)).flatten()

        #将x,y的坐标更新到anchor里面去,输出anchor的shape为【anchor_num*size*size,4]，坐标为中心点形式
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        # [n,4*a,h,w]-->[4*a,h,w,n]-->[4,a*h*w*n]
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()
        # delta的shape为【4,a*h*w*n],n=1,anchor的shape为【a*h*w,4】
        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]

        # 返回的delta就是在搜索区域图像上bbox,shape为【4，a*h*w*n]
        return delta

    def _convert_score(self, score):
        # [n,2*a,h,w]-->[2*a,h,w,n]-->[2,a*h*w*n]-->[a*h*w*n,2]
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()  # 通道1表示是目标的概率，输出>[a*h*w*n]
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        # cx,cy,w,h都是对应图像原始分辨率的，boundary就是输入图像的高度宽度
        # 剪切的作用是为了让输出边界框中心不要出边界，同时边界框的大小不要超过图像本身大小，同时也别小于10*10
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)           #加上上下文信息区域认为是模板
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))      #计算均值填充padding

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)

        #用单个图像初始化前seq_in_len帧，作为模板
        for i in range(self.model.grus.seq_in_len):
            self.template_idx=i
            self.model.gru_template(z_crop,self.template_idx)                      #跟踪的时候先把模板分支的前向存储下来


    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        #计算模板图的尺寸
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)    #self.size 加上上文章信息，才认为是模板区域
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z

        #根据模板图的尺寸以及网络两个分支输入图像尺寸的比例，计算搜索区域的尺寸
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)              #注意这个track是网络定义里面的track，只是做搜索区域的特征提取和相关层rpn层计算

        score = self._convert_score(outputs['cls'])     #按照softmax计算概率
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)        #按照anchor将网络预测结果dx,dy,dw,dh转化到实际坐标x,y,w,h

        def change(r):          #如果r>1,取值r,如果r<1,则r取1/r ,只取他们相对变化量，而不管两者具体谁大谁小
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty,计算当前尺寸/上一帧目标的尺寸的比例r,通过change调整，如果r>1,取值r,如果r<1,则r取1/r，作为尺度惩罚项
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty 比例因子惩罚项
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE    #hannig窗口惩罚
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                'bbox': bbox,
                'best_score': best_score
               }
