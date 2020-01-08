from os.path import join, isdir
from os import listdir, mkdir, makedirs
from os import system
import cv2
import numpy as np
import glob
from concurrent import futures
import sys
import time
import json

Run_In_Terminal = False

got_base_path ='/home/rainzsy/datasets/got10k/'
sub_sets = sorted({'train'})



# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):


    # 通过下面的方式会对目标区域进行缩放，当时长宽上的缩放比例保持一致，也就是说目标不会变形，因为最后的模板是正方形的
    # 对于长方形的部分，短边就用背景来填补，也就是说对于狭长的bbox并不友好，会引入较多的背景信息

    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]           #x1,y1,x2,y2--->cx,cy,w,h
    target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)           #上下文信息的宽度 wc = w+0.5*(w+h)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)                                          #平衡高度和宽度之后的目标的尺寸，对应图像分辨率
    scale_z = exemplar_size / s_z                                       #目标区域放大（或缩小）到127*127需要缩放的比例
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad                                                 #搜索区域的大小，对应图像分辨率

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)  #将含有一部分背景的区域当做目标，并缩放到127*127
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)   #将包含更多背景的区域当做搜索区域并缩放到511*511
    return z, x             #z是模板，大小为127*127，！！！！（不完全是bbox，还包括bbox外面0.5倍的背景，这个叫上下文信息。。。。），x是搜搜区域，大小为511*511





def crop_video(base_path,sub_set, video_name, crop_path, instanc_size,snippets):
    # print(video_name)
    video_crop_base_path = join(crop_path, sub_set, video_name)
    video_path = join(base_path,video_name)
    if not isdir(video_crop_base_path):
        makedirs(video_crop_base_path)

    for trackid,snippet in snippets.items():
        for image_name,bbox in snippet.items():
            image_name =int(image_name)
            image_name = "{:08d}".format(image_name)        #got10k中图像序号的命名规范是按照8位来命名的

            img_path=join(video_path,image_name+'.jpg')
            im = cv2.imread(img_path)
            avg_chans = np.mean(im, axis=(0, 1))  # 先计算图像的均值
            # 将bbox作为初始区域，包含0.5倍的上下文信息的区域当做模板，再扩展一部分当做搜索区域
            z, x = crop_like_SiamFC(im, bbox, instanc_size=instanc_size, padding=avg_chans)
            # cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(int(image_name), int(trackid))), z)
            cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(int(image_name), int(trackid))), x)





def main(instanc_size=511, num_threads=24):

    crop_path = got_base_path+'/crop{:d}'.format(instanc_size)

    if not isdir(crop_path): mkdir(crop_path)

    for sub_set in sub_sets:
        got10k = json.load(open('{}.json'.format(sub_set), 'r'))
        n_videos = len(got10k)
        # concurrent.futures 模块提供了一个高水平的接口用于异步执行调用,有线程实现和进程实现两种方式
        #这里采用进程方式执行异步并行调用

        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            #submit(fn, *args, **kwargs)调度可调用的fn，作为fn(args,  kwargs)执行，并返回一个表示可调用的执行的Future对象(线程池对象)
            #这里调用crop_video这个函数，输入数据为 got_base_path,sub_set, video, crop_path, instanc_size,got10k[video]
            fs = [executor.submit(crop_video, got_base_path,sub_set, video, crop_path, instanc_size,got10k[video]) for video in got10k]

            #将所有的线程池处理进度综合到一块并打印进度信息
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, n_videos, prefix=sub_set, suffix='Done ', barLength=40)


if __name__ == '__main__':
    since = time.time()

    if Run_In_Terminal:
        main(int(sys.argv[1]), int(sys.argv[2]))
    else:
        # system("bash ./create_link.sh")
        main(511,12)
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
