
import cv2
import torch


def draw_rect(image_batch,boxes):
    '''

    :param image_batch:
    :param boxes: shape:128*4,
    :param neg:
    :return:
    '''
    color = (0, 0, 255)
    image_batch=image_batch.permute(0,2,3,1).contiguous().numpy()
    for b in range(image_batch.shape[0]):
        # if neg[b]==0:       #如果不是负样本才画框
        for  n in range(boxes.shape[1]):        #第n个bbox
            try:
                cv2.rectangle(image_batch[b,...],pt1=(boxes[b,n,0], boxes[b,n,1]),pt2=(boxes[b,n,2], boxes[b,n,3]),color=color,thickness=2)
            except:
                pass

        image_batch[b,...] = cv2.cvtColor(image_batch[b,...], cv2.COLOR_BGR2RGB)

    image_batch = torch.from_numpy(image_batch.transpose(0,3,1,2)).float()
    return image_batch



# boxes=self._convert_bbox(loc2_d,anchor_center2.detach())[indexx,:,poss]
# image_show = draw_rect(data['search'][:self.show_num, ...], boxes[:self.show_num, ...],
#                        data['neg'][:self.show_num, ...])
# image_show = vutils.make_grid(image_show, normalize=True, scale_each=True)
# show['Image_second']=image_show
