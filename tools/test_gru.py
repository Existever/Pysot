# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

from pysot.models.backbone.alexnet import AlexNet


root_dir ='/home/rainzsy/projects/Pytorch/Pysot/'
datasets_root ='/home/rainzsy/datasets/vot/'


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', default='VOT2018',
                    type=str, help='datasets')
parser.add_argument('--config', default=root_dir+'experiments/siamrpn_alex_dwxcorr_16gpu/config_gru.yaml',
                    type=str,   help='config file')
parser.add_argument('--snapshot', default=root_dir+'experiments/siamrpn_alex_dwxcorr_16gpu/gru_snapshot/checkpoint_e37_fitune.pth',
                    type=str,    help='snapshot of models to eval')
parser.add_argument('--video', default='',
                    type=str,   help='eval one special video')
parser.add_argument('--vis', default=True,
                    action='store_true', help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)



# parser = argparse.ArgumentParser(description='siamrpn tracking')
# parser.add_argument('--dataset', default='VOT2018',
#                     type=str, help='datasets')
# parser.add_argument('--config', default=root_dir+'experiments/siamrpn_r50_l234_dwxcorr/config.yaml',
#                     type=str,   help='config file')
# parser.add_argument('--snapshot', default=root_dir+'experiments/siamrpn_r50_l234_dwxcorr/model.pth',
#                     type=str,    help='snapshot of models to eval')
# parser.add_argument('--video', default='',
#                     type=str,   help='eval one special video')
# parser.add_argument('--vis', default=True,
#                     action='store_true', help='whether visualzie result')
# args = parser.parse_args()
#
# torch.set_num_threads(1)


def save_backbone(siamese):
    alexnet =AlexNet()
    alexnet_state_dict = alexnet.state_dict()
    siamese_state_dict =siamese.state_dict()
    for key in siamese_state_dict:
        print(key)

    for key in alexnet_state_dict:
        alexnet_state_dict[key]=siamese_state_dict['backbone.'+key]
        print(key)

    torch.save(alexnet_state_dict,"siamese_alexnet_backbone.pth")




def save_siamese_rpn():
    # load config

    rpn_path = root_dir + 'experiments/siamrpn_alex_dwxcorr_16gpu/pre_train/checkpoint_e45.pth'
    gru_rpn = root_dir + 'experiments/siamrpn_alex_dwxcorr_16gpu/config.yaml'
    cfg.merge_from_file(gru_rpn)
    # create model
    model_rpn = ModelBuilder()
    model_rpn = load_pretrain(model_rpn, rpn_path).cuda().eval()


    gru_path = root_dir + 'experiments/siamrpn_alex_dwxcorr_16gpu/gru_snapshot/gru_10.pth'
    gru_cfg=root_dir + 'experiments/siamrpn_alex_dwxcorr_16gpu/config_gru.yaml'
    cfg.merge_from_file(gru_cfg)
    # create model
    model_gru= ModelBuilder()
    model_gru = load_pretrain(model_gru, gru_path).cuda().eval()






    for key ,item in model_gru.named_parameters():

        # print(key.find("grus"))
        print(key,item.shape)

    for key, item in model_rpn.named_parameters():
        # print(key.find("grus"))
        print(key, item.shape)

    model_gru_dict = model_gru.state_dict()
    model_rpn_dict = model_rpn.state_dict()

    for key in model_gru_dict:

        if key.find("grus") !=-1:
            print("fix:",key)

        else:
            print("change:",key)
            model_gru_dict[key]=model_rpn_dict[key]





    # name_map={}
    # model_legacy_dict = model_legacy.state_dict()
    # model_alexnet_dict = model_alexnet.state_dict()
    # for para1,para2 in zip(model_legacy.named_parameters(),model_alexnet.named_parameters()):
    #     # print(para1[0],para1[1].shape)
    #     print(para1[0])
    #     print(para2[0])
    #     print(para1[1].shape)
    #     print(para2[1].shape)
    #     print("--"*40)
    #     # print("['{}'--->'{}']".format(para1[0], para2[0]),para1[1].shape, para2[1].shape)
    #     name_map[para1[0]]=para2[0]
    # print(name_map)
    #
    #
    # for key,val in name_map.items():
    #     model_alexnet_dict[val]=model_legacy_dict[key]

    torch.save(model_gru_dict, "siamese_gru10_rpn45.pth")





def main():
    # load config
    # save_siamese_rpn()
    cfg.merge_from_file(args.config)



    cur_dir = os.path.dirname(os.path.realpath(__file__))
    # dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)
    dataset_root = datasets_root+ args.dataset

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()






    # save_backbone(model)

    # build tracker
    tracker = build_tracker(model)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0

    #multi-pass tracking,跟踪丢失后重新初始化的测试方法
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0

            # pred_bboxes包含两种类型的数据，类型1：整型数据，有1,2，0,三个值，分别表示跟踪开始，跟踪结束（丢失），跟踪丢失之后，间隔帧的占位符
            # 类型2：浮点类型的bbox,也就是跟踪结果
            pred_bboxes = []

            gru_seq_len=tracker.model.grus.seq_in_len
            video_len =len(video)

            for idx, (img, gt_bbox) in enumerate(video):




                if len(gt_bbox) == 4:     #如果gt是【x，y,w,h】的方式，转化为8个坐标信息（x1,y1,x2,y2,x3,y3,x4,y4）
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()

                #跟踪初始化
                if idx == frame_counter:#   跟踪第一帧初始化
                    idxs = list(map(lambda x, y: x + y, [idx] * gru_seq_len,
                                    list(range(0, gru_seq_len))))  # 取出idx后面的gru_seq_len个序列的索引号
                    idxs = list(map(lambda x: min(x, video_len - 1), idxs))  # 避免索引号越界

                    tracker.template_idx=0          #模板初始化的第一帧
                    for k in idxs:
                        init_img, init_gt_bbox = video[k]           #连续gru_seq_len帧初始化
                        #init_img, init_gt_bbox =video[idxs[0]]     #只用一帧作为初始化参数

                        cx, cy, w, h = get_axis_aligned_bbox(np.array(init_gt_bbox))    #将倾斜框4个点坐标，转化为bbox,x,y为中心点形式(cx,cy,w,h)
                        init_gt_bbox = [cx-(w-1)/2, cy-(h-1)/2, w, h]                  #x,y,中心点形式，转化为左上角形式

                        tracker.init_gru(init_img, init_gt_bbox,k)

                    if k==0:
                        pred_bbox = init_gt_bbox
                        pred_bboxes.append(1)

                #持续的后续跟踪
                elif idx > frame_counter:
                    outputs = tracker.track(img)                #对于下面的帧
                    pred_bbox = outputs['bbox']

                    #只有输出概率很高的时候才更新模板
                    if outputs['best_score']>0.95:
                        tracker.init_gru(img, pred_bbox, idx)


                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))

                    #查看初始化后的第一帧检测iou和score之间的关系
                    # if tracker.template_idx==4:
                    #     print("{:3.2f}\t{:3.2f}".format(overlap,outputs['best_score']))

                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)



                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()

                #绘制输出框，gt和mask都按照多边形来绘制，跟踪的bbox按照矩形来绘制
                if args.vis and idx > frame_counter:
                    #绘制多边形的gt
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                    #绘制siamesemask输出的多边形
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))], True, (0, 255, 255), 3)
                    #绘制输出矩形框
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)

                    #添加图像标注，帧号和丢失次数
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join('results', args.dataset, model_name,'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            #结果路径的构成： ./results/VOT2018/model/baseline/ants1/ants1_001.txt
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))

            #pred_bboxes包含两种类型的数据，类型1：整型数据，有1,2，0,三个值，分别表示跟踪开始，跟踪结束（丢失），跟踪丢失之后，间隔帧的占位符
            # 类型2：浮点类型的bbox,也就是跟踪结果
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):              #整数代表开始，或者有丢失
                        f.write("{:d}\n".format(x))
                    else:                               #浮点数才是bbox
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))

    #oetracking,跟踪丢失后不再重新初始化的测试方法
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:

                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join('results', args.dataset, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
