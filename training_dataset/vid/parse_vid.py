from os.path import join
from os import listdir
import json
import glob
import xml.etree.ElementTree as ET

VID_base_path ='./ILSVRC2015'
ann_base_path = join(VID_base_path, 'Annotations/VID/train/')
img_base_path = join(VID_base_path, 'Data/VID/train/')
sub_sets = sorted({'a'})



# VID_base_path = './ILSVRC2015'
# ann_base_path = join(VID_base_path, 'Annotations/VID/train/')
# img_base_path = join(VID_base_path, 'Data/VID/train/')
# sub_sets = sorted({'a', 'b', 'c', 'd', 'e'})

vid = []
for sub_set in sub_sets:
    sub_set_base_path = join(ann_base_path, sub_set)   #子数据集合基础路径，下面分多个视频序列的文件夹例如其中一个为（ILSVRC2015_train_00000000）  './ILSVRC2015/Annotations/VID/train/a',
    videos = sorted(listdir(sub_set_base_path))         #列出视频序列的文件夹名字
    s = []
    for vi, video in enumerate(videos):
        print('subset: {} video id: {:04d} / {:04d}'.format(sub_set, vi, len(videos)))
        v = dict()
        v['base_path'] = join(sub_set, video)           #存储到json文件中的视频路径为部分路径， 'a/ILSVRC2015_train_00000000'
        v['frame'] = []
        video_base_path = join(sub_set_base_path, video)#某个视频的全部路径，因为要找出其中的图片'./ILSVRC2015/Annotations/VID/train/a/ILSVRC2015_train_00000000'
        xmls = sorted(glob.glob(join(video_base_path, '*.xml')))
        for xml in xmls:
            f = dict()          #一个file或者说frame包含的信息
            xmltree = ET.parse(xml)
            size = xmltree.findall('size')[0]
            frame_sz = [int(it.text) for it in size]            #图像尺寸
            objects = xmltree.findall('object')
            objs = []
            for object_iter in objects:
                trackid = int(object_iter.find('trackid').text)     #跟踪目标的id
                name = (object_iter.find('name')).text              #目标的类别
                bndbox = object_iter.find('bndbox')                 #目标的bbox
                occluded = int(object_iter.find('occluded').text)   #是否有遮挡
                o = dict()                      #某一个目标包含的信息
                o['c'] = name
                o['bbox'] = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                             int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
                o['trackid'] = trackid
                o['occ'] = occluded
                objs.append(o)      #一帧包含多个目标的信息

            f['frame_sz'] = frame_sz
            f['img_path'] = xml.split('/')[-1].replace('xml', 'JPEG')       #xml是某个标签的名字，例如000001.xml--->000001.JPEG
            f['objs'] = objs

            #一个视频包含多帧的信息
            v['frame'].append(f)
        #一个子数据集包含多个视频的数据
        s.append(v)
    #一个vid数据集包含多个子数据集合
    vid.append(s)
print('save json (raw vid info), please wait 1 min~')
json.dump(vid, open('vid_mini.json', 'w'), indent=4, sort_keys=True)
print('done!')
