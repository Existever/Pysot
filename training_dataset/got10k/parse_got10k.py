from __future__ import absolute_import, print_function

import os
import glob
import numpy as np
import six
import json
import numpy as np

root_dir = '/media/solanliu/disk/zsy/datasets/got10k/'
subset = 'train'


class GOT10k(object):
    r"""`GOT-10K <http://got-10k.aitestunion.com//>`_ Dataset.

    Publication:
        ``GOT-10k: A Large High-Diversity Benchmark for Generic Object
        Tracking in the Wild``, L. Huang, X. Zhao and K. Huang, ArXiv 2018.

    Args:
        root_dir (string): Root directory of dataset where ``train``,
            ``val`` and ``test`` folders exist.
        subset (string, optional): Specify ``train``, ``val`` or ``test``
            subset of GOT-10k.
        return_meta (string, optional): If True, returns ``meta``
            of each sequence in ``__getitem__`` function, otherwise
            only returns ``img_files`` and ``anno``.
        list_file (string, optional): If provided, only read sequences
            specified by the file instead of all sequences in the subset.
    """

    def __init__(self, root_dir, subset='test', return_meta=False,
                 list_file=None, check_integrity=True):
        super(GOT10k, self).__init__()
        assert subset in ['train', 'val', 'test'], 'Unknown subset.'
        self.root_dir = root_dir
        self.subset = subset
        self.return_meta = False if subset == 'test' else return_meta

        if list_file is None:
            list_file = os.path.join(root_dir, subset, 'list.txt')
        if check_integrity:
            self._check_integrity(root_dir, subset, list_file)

        with open(list_file, 'r') as f:
            self.seq_names = f.read().strip().split('\n')
        self.seq_dirs = [os.path.join(root_dir, subset, s)
                         for s in self.seq_names]
        self.anno_files = [os.path.join(d, 'groundtruth.txt')
                           for d in self.seq_dirs]

    def __getitem__(self, index):
        r"""
        Args:
            index (integer or string): Index or name of a sequence.

        Returns:
            tuple: (img_files, anno) if ``return_meta`` is False, otherwise
                (img_files, anno, meta), where ``img_files`` is a list of
                file names, ``anno`` is a N x 4 (rectangles) numpy array, while
                ``meta`` is a dict contains meta information about the sequence.
        """
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(os.path.join(
            self.seq_dirs[index], '*.jpg')))
        anno = np.loadtxt(self.anno_files[index], delimiter=',')

        if self.subset == 'test' and anno.ndim == 1:
            assert len(anno) == 4
            anno = anno[np.newaxis, :]
        else:
            assert len(img_files) == len(anno)

        if self.return_meta:
            video_name=self.seq_names[index]
            meta = self._fetch_meta(self.seq_dirs[index])
            return video_name,img_files, anno, meta
        else:
            return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _check_integrity(self, root_dir, subset, list_file=None):
        assert subset in ['train', 'val', 'test']
        if list_file is None:
            list_file = os.path.join(root_dir, subset, 'list.txt')

        if os.path.isfile(list_file):
            with open(list_file, 'r') as f:
                seq_names = f.read().strip().split('\n')

            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, subset, seq_name)
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            # dataset not exists
            raise Exception('Dataset not found or corrupted.')

    def _fetch_meta(self, seq_dir):
        # meta information
        meta_file = os.path.join(seq_dir, 'meta_info.ini')
        with open(meta_file) as f:
            meta = f.read().strip().split('\n')[1:]
        meta = [line.split(': ') for line in meta]
        meta = {line[0]: line[1] for line in meta}

        # attributes
        attributes = ['cover', 'absence', 'cut_by_image']
        for att in attributes:
            meta[att] = np.loadtxt(os.path.join(seq_dir, att + '.label'))

        return meta



if __name__ =='__main__':

    list_file = os.path.join(root_dir,subset,'list.txt')      #视频序列的名字构成的list
    got10k_dataset=GOT10k(root_dir,subset,True,list_file)
    got10k = []
    #仿照vid的数据格式组织got10k,一个数据集包含多个视频，一个视频包含多帧，一帧中包含多个目标，一个目标一个跟踪id和gt_bbox
    #由于got10k,是单目标跟踪，一个数据集包含多个视频，一个视频包含多帧，一帧包含一个目标，一个目标一个gt_bbox
    for vi,video_info in enumerate(got10k_dataset):
        video_name,img_files, gt_bbox, meta =video_info
        if vi %1000==0:
            print(vi,video_name)

        v = dict()
        v['base_path'] = os.path.join(subset,video_name)           #存储到json文件中的视频路径为部分路径， 'a/ILSVRC2015_train_00000000'
        v['frame'] = []
        frame_sz = meta['resolution'].strip('(').strip(')').split(',')
        frame_sz = [int(s) for s in frame_sz]
        class_name = meta['major_class']

        for f_idx  in range(len(img_files)):
            f = dict()  # 一个file或者说frame包含的信息
            f['frame_sz'] = frame_sz
            f['img_path'] = img_files[f_idx].split('/')[-1]  # --->00000001.jpg

            objs = []
            for b_idx in range(1):      #got是单目标跟踪，一张图中bbox只有一个
                bbox=gt_bbox[f_idx]
                cover = meta['cover'][f_idx]        #可见性比例
                cut = meta['cut_by_image'][f_idx]   #目标是否被图像切分
                absence=meta['absence'][f_idx]      #目标是否出视场
                o = dict()                      #某一个目标包含的信息
                o['c'] = class_name
                o['bbox'] = [ bbox[0], bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]   #got坐标类型为【xmin,ymin,w,h】的形式转化为【xmin,ymin,xmax,ymax】
                o['trackid'] = 0            #got是单目标跟踪，一张图中bbox只有目标
                o['cover'] = cover
                o['cut'] = cut
                o['absence'] = absence

                # if cover < 4:
                #     print('*' * 40, img_files[f_idx], absence, cover)

                objs.append(o)      #一帧可以包含多个目标的信息，但是这里只要一个
            f['objs'] = objs
            #一个视频包含多帧的信息
            v['frame'].append(f)
        #一个数据集包含多个视频的数据
        got10k.append(v)

    print('save json (raw vid info), please wait 1 min~')
    json.dump(got10k, open('got10k_{}.json'.format(subset), 'w'), indent=4, sort_keys=True)
    print('done!')