# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2

from pysot.utils.bbox import corner2center, \
        Center, center2corner, Corner


class Augmentation:
    def __init__(self, shift, scale, blur, flip, color):
        self.shift = shift
        self.scale = scale
        self.blur = blur
        self.flip = flip
        self.color = color
        self.rgbVar = np.array(
            [[-0.55919361,  0.98062831, - 0.41940627],
             [1.72091413,  0.19879334, - 1.82968581],
             [4.64467907,  4.73710203, 4.88324118]], dtype=np.float32)

    @staticmethod
    def random():
        return np.random.random() * 2 - 1.0

    def _crop_roi(self, image, bbox, out_sz, padding=(0, 0, 0)):
        '''
        仿射变换参数矩阵：s代表缩放因子，r代表逆时针旋转角度， tx,ty代表平移https://www.cnblogs.com/shine-lee/p/10950963.html
        [  s*cos(r)  -s*sin(r)   tx]  [x1]     [x2]
        [  s*sin(r)   s*cos(r)   ty]  [y1] =   [y2]
                                      [1 ]     [s]
        :param image:
        :param bbox:
        :param out_sz:
        :param padding:
        :return:
        '''
        bbox = [float(x) for x in bbox]
        a = (out_sz-1) / (bbox[2]-bbox[0])     #x方向的缩放因子
        b = (out_sz-1) / (bbox[3]-bbox[1])     #y方向的缩放因子
        c = -a * bbox[0]                       #x方向的平移
        d = -b * bbox[1]                       #y方向的平移
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _blur_aug(self, image):
        def rand_kernel():
            sizes = np.arange(5, 46, 2)
            size = np.random.choice(sizes)
            kernel = np.zeros((size, size))
            c = int(size/2)
            wx = np.random.random()
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1-wx)
            return kernel
        kernel = rand_kernel()
        image = cv2.filter2D(image, -1, kernel)
        return image

    def _color_aug(self, image):   #按照给定的rgb方差生成偏置，减去偏置项目
        offset = np.dot(self.rgbVar, np.random.randn(3, 1))
        offset = offset[::-1]  # bgr 2 rgb
        offset = offset.reshape(3)
        image = image - offset
        return image

    def _gray_aug(self, image):
        grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)
        return image

    def _shift_scale_aug(self, image, bbox, crop_bbox, size):
        ''' 对具有上下文信息的gt bbox进行位移和缩放调整，然后输出的bbox，和对应的图像区域
        :param image:
        :param bbox:  带有上下文信息的box（gt值）
        :param crop_bbox: 要crop的bbox位置信息
        :param size: 期望crop出来的区域尺寸，网络输入时模板大小127*127，或者搜索区域大小255*255
        :return:返回的图像，是按照增强后的crop_box扣取出的roi图像区域，返回的bbox是gt信息也做相应调整后并转化到crop图像坐标系下的位置信息
        '''
        im_h, im_w = image.shape[:2]

        # adjust crop bounding box
        crop_bbox_center = corner2center(crop_bbox)          #对要crop输出的box进行大小调整和位移调整
        if self.scale:
            scale_x = (1.0 + Augmentation.random() * self.scale)
            scale_y = (1.0 + Augmentation.random() * self.scale)
            h, w = crop_bbox_center.h, crop_bbox_center.w
            scale_x = min(scale_x, float(im_w) / w)         #对要crop输出的box的w,h进行调整，取最小值是为了上搜索区域w,h不要超过图像区域
            scale_y = min(scale_y, float(im_h) / h)
            crop_bbox_center = Center(crop_bbox_center.x,
                                      crop_bbox_center.y,
                                      crop_bbox_center.w * scale_x,
                                      crop_bbox_center.h * scale_y)

        crop_bbox = center2corner(crop_bbox_center)
        if self.shift:
            sx = Augmentation.random() * self.shift             #siamese rpn++ 论文中讨论了shift最大范围的时候能够一定程度上解决网络学习过程中的位置偏见问题
            sy = Augmentation.random() * self.shift

            x1, y1, x2, y2 = crop_bbox

            sx = max(-x1, min(im_w - 1 - x2, sx))   #min(im_w - 1 - x2, sx) 保证x2+sx不会超出图像右边界，也就是即使平移搜索区域，右边也不要超出右边图像边界，max(-x1,xxx)是保证x1+xxx不会小鱼0，也就是即使平移搜索区域，左边也不会超出左边图像边界
            sy = max(-y1, min(im_h - 1 - y2, sy))

            crop_bbox = Corner(x1 + sx, y1 + sy, x2 + sx, y2 + sy)

        # adjust target bounding box  要crop的box的变换上面已经确定，这里需要将他的gt信息也同样做调整
        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = Corner(bbox.x1 - x1, bbox.y1 - y1,       #以要crop输出的box的左上角为参考点，计算bbox新的坐标，也就是相应得修改gt的信息，与要crop的内容保持一致
                      bbox.x2 - x1, bbox.y2 - y1)

        if self.scale:
            bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y,
                          bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_roi(image, crop_bbox, size)     #扣取出要crop的区域
        return image, bbox

    def _flip_aug(self, image, bbox):
        '''
        :param image:
        :param bbox:
        :return:  左右翻转
        '''
        image = cv2.flip(image, 1)
        width = image.shape[1]
        bbox = Corner(width - 1 - bbox.x2, bbox.y1,
                      width - 1 - bbox.x1, bbox.y2)
        return image, bbox

    def __call__(self, image, bbox, size, gray=False):
        '''
        :param image: crop后的图像，大小511*511，模板图像已经对齐到图像中心，
        :param bbox: 带有上下文信息的box大小
        :param size: 网络输入时模板大小127*127，或者搜索区域大小255*255
        :param gray: 是否进行灰度化
        :return:
        '''
        shape = image.shape                #固定大小511*511
        crop_bbox = center2corner(Center(shape[0]//2, shape[1]//2,              #要从image中抠出搜索区域，这里计算出模板在图中左上角和右下角的坐标
                                         size-1, size-1))
        # gray augmentation（如果随机选择过程要进行灰度花，则先将彩色图像转化为灰度，在从灰度转化为3通道“彩图”）
        if gray:
            image = self._gray_aug(image)

        # shift scale augmentation
        image, bbox = self._shift_scale_aug(image, bbox, crop_bbox, size)

        # color augmentation
        if self.color > np.random.random():
            image = self._color_aug(image)

        # blur augmentation
        if self.blur > np.random.random():
            image = self._blur_aug(image)

        # flip augmentation
        if self.flip and self.flip > np.random.random():
            image, bbox = self._flip_aug(image, bbox)
        return image, bbox
