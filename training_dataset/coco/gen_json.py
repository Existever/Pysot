from pycocotools.coco import COCO
from os.path import join
import json

#
# dataDir = '.'

dataDir = '/media/rainzsy/00024268000F00F7/coco'


'''
#.json的生成标签文件,生成的dataset格式为

dataset=
【
    'val2017/000000397133'：[                                    #视频id
                                “00":                            #跟踪目标id
                                    {
                                    "000000":[x1,y1,x2,y2]       #key:图像帧号,value :bbox
                                    }
                                 “01":
                                    {
                                    "000001":[x1,y1,x2,y2]
                                    }
                            ]
   #同理类推... 
    'val2017/000000397134':
    ...
                             
】
'''


for dataType in ['val2017', 'train2017']:
    dataset = dict()
    annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)
    coco = COCO(annFile)
    n_imgs = len(coco.imgs)
    for n, img_id in enumerate(coco.imgs):
        print('subset: {} image id: {:04d} / {:04d}'.format(dataType, n, n_imgs))
        img = coco.loadImgs(img_id)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        video_crop_base_path = join(dataType, img['file_name'].split('/')[-1].split('.')[0])
        
        if len(anns) > 0:
            dataset[video_crop_base_path] = dict()        

        for trackid, ann in enumerate(anns):
            rect = ann['bbox']
            c = ann['category_id']
            bbox = [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]
            if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
                continue
            dataset[video_crop_base_path]['{:02d}'.format(trackid)] = {'000000': bbox}

    print('save json (dataset), please wait 20 seconds~')
    json.dump(dataset, open(dataDir+'/{}.json'.format(dataType), 'w'), indent=4, sort_keys=True)


    print('done!')

