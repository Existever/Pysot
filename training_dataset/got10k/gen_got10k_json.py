from os.path import join
from os import listdir
import json
import numpy as np

#先要运行parse_got10k.py生成got10k_{}.json



subset = 'train'
got10k = json.load(open('got10k_{}.json'.format(subset), 'r'))


def check_size(frame_sz, bbox):
    #检查目标尺寸相对于图像尺寸过大或者过小的bbox
    min_ratio = 0.1
    max_ratio = 0.75
    # only accept objects >10% and <75% of the total frame
    area_ratio = np.sqrt((bbox[2]-bbox[0])*(bbox[3]-bbox[1])/float(np.prod(frame_sz)))
    ok = (area_ratio > min_ratio) and (area_ratio < max_ratio)
    return ok


def check_borders(frame_sz, bbox):
    #检查目标bbox出边界的情况
    dist_from_border = 0.05 * (bbox[2] - bbox[0] + bbox[3] - bbox[1])/2
    ok = (bbox[0] > dist_from_border) and (bbox[1] > dist_from_border) and \
         ((frame_sz[0] - bbox[2]) > dist_from_border) and \
         ((frame_sz[1] - bbox[3]) > dist_from_border)
    return ok


snippets = dict()
n_snippets = 0      #视频片段的个数，注意一个视频由于目标长期出视场可能会被切分为多个小的片段
n_videos = 0

for video in got10k:
    n_videos += 1
    frames = video['frame']
    id_set = []
    id_frames = [[]] * 60  # at most 60 objects,一个视频序列中最多只要60个跟踪目标
    for f, frame in enumerate(frames):
        objs = frame['objs']
        frame_sz = list(frame['frame_sz'])
        for obj in objs:
            trackid = obj['trackid']
            bbox = obj['bbox']

            #可见性小于4的序列跳过，那么一个视频可能会被分割为多个小的片段
            if obj['cover']<4:
                # print("cover<4:",video['base_path'],frame['img_path'])
                continue

            # if not(check_size(frame_sz, bbox) and check_borders(frame_sz, bbox)):
            #     print("out of border:", video['base_path'], frame['img_path'])
            #     continue



            if trackid not in id_set:
                id_set.append(trackid)
                id_frames[trackid] = []
            id_frames[trackid].append(f)
    if len(id_set) > 0:
        snippets[video['base_path']] = dict()

    for selected in id_set:
        frame_ids = sorted(id_frames[selected])     #这个视频序列中的一个跟踪目标

        #由于视频序列中某个目标长期出视场，则跟踪目标id就会丢失，也就是说frames_ids会被切分成若干个小的片段，
        # 例如【1,2,3....50,51,124,125,126,....170,171】这样的结果，下面就是要找出这样的小片段snippet
        #np.diff(frame_ids)沿着指定轴计算第N维的离散差值,（np.where(np.diff(frame_ids) > 1)[0] 找到相邻帧号
        # 大于1的位置的索引，在该位置后面一个位置（加1操作）利用split按照这个索引切分为若干个小的片段
        sequences = np.split(frame_ids, np.array(np.where(np.diff(frame_ids) > 1)[0]) + 1)
        sequences = [s for s in sequences if len(s) > 1]  # remove isolated frame. 剔除只有单帧的小片段
        for id,seq in enumerate(sequences):
            # 对于每个小片段当做一个有效的跟踪视频序列，提取其中的标签信息，生成字典
            snippet = dict()
            for frame_id in seq:
                frame = frames[frame_id]
                for obj in frame['objs']:
                    if obj['trackid'] == selected:
                        o = obj
                        continue
                frame_num=int(frame['img_path'].split('.')[0])
                frame_num = "{:06d}".format(frame_num)      #按照pysot中读取图像数据命名规范，只有6位数字
                snippet[frame_num] = o['bbox']

            #snippets中存放所有视频片段的信息，以视频的部分路径作为id,snippets的组织方式是
            #一个视频有多个跟踪目标，一个跟踪目标在多个帧中连续出现，每个跟踪目标按照trackid为key,多帧标注信息为value
            snippets[video['base_path']]['{:02d}'.format(id)] = snippet
            n_snippets += 1
    print('video: {:d} snippets_num: {:d}'.format(n_videos, n_snippets))

#snippets字典含有三级信息，【视频部分路径】【跟踪id】【图像帧号】,通过这三级字典就可以访问到某个视频中某个跟踪目标在
#某一帧中的bbox,其中bbox按照【x1,y1,x2,y2】的方式表示，注意这些bbox都是相对于原图来说的，和crop为511的图没有任何关系
#那又是通过什么方式知道目标在crop511图中的位置信息的呢？

# 因为输入的图片的数据已经是crop为511*511的图像了，模板的size(127*127)，并放在图像正中心，如果模板不做数据增强的话，这一步
# 直接从中心扣取127*127的区域就是模板（其实是包含0.5倍的上下文图像内容的），扣取255*255的区域就是搜索区域，由于在crop511的时候
# 目标的长宽是等比例扣取的，由于要求模板是个正方形，对于长方形的bbox的短边就用背景填充，而目标在crop511图像的宽高信息在就是通过
# 这里提供的在原始图像下的bbox里的w,h通过相同规则等比例计算得来的，从而确定目标在crop511中的具体位置


train = {k:v for (k,v) in snippets.items() if 'Train' in k}
val = {k:v for (k,v) in snippets.items() if 'Val' in k}

json.dump(train, open('train.json', 'w'), indent=4, sort_keys=True)
json.dump(val, open('val.json', 'w'), indent=4, sort_keys=True)
print('done!')
