import os
import numpy as np
from PIL import Image
from utils import IOUMetric


if __name__ == '__main__':
    mIOU = IOUMetric(num_classes=21)
    root_dir = '/home/ubuntu/workshops/datasets/voc12/VOCdevkit/VOC2012/'

    pred_dir = './exp/labels'
    gt_dir = root_dir + 'SegmentationClass/'
    ids = [i.strip() for i in open(root_dir + 'ImageSets/Segmentation/val.txt') if not i.strip() == '']

    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0.5, 0.5, 0), (0, 0, 0.5), (0.5, 0, 0.5), (0, 0.5, 0.5),
            (0.5, 0.5, 0.5), (0.25, 0, 0), (0.75, 0, 0), (0.25, 0.5, 0), (0.75, 0.5, 0), (0.25, 0, 0.5),
            (0.75, 0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5), (0, 0.25, 0), (0.5, 0.25, 0), (0, 0.75, 0),
            (0.5, 0.75, 0), (0, 0.25, 0.5)]
    values = [i for i in range(21)]
    color2val = dict(zip(colormap, values))
    # print(color2val)

    import time
    st = time.time()
    for ind, img_id in enumerate(ids):
        img_path = os.path.join(gt_dir, img_id+'.png')
        pred_img_path = os.path.join(pred_dir, img_id+'.png')

        gt = Image.open(img_path)
        w, h = gt.size[0], gt.size[1]
        gt = np.array(gt, dtype=np.int32)   # shape = [h, w], 0-20 is classes, 255 is ingore boundary
        
        pred = Image.open(pred_img_path)
        pred = pred.crop((0, 0, w, h))
        pred = np.array(pred, dtype=np.int32)   # shape = [h, w]
        mIOU.add_batch(pred, gt)
        # print(img_id, ind)

    acc, acc_cls, iou, miou, fwavacc = mIOU.evaluate()
    print(acc, acc_cls, iou, miou, fwavacc)
    print('mIOU = %s, time = %s s' % (miou, str(time.time() - st)))
