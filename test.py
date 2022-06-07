import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from PIL import Image
import time
from time import strftime, localtime
import cv2
import argparse
import os
from datasets import VOCDataset
from nets import vgg
import utils
import crf
from tqdm import tqdm
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]

pred_dir_path = './exp/'


# class palette for test
palette = []
for i in range(256):
    palette.extend((i,i,i))
palette[:3*21] = np.array([[0, 0, 0],
                        [128, 0, 0],
                        [0, 128, 0],
                        [128, 128, 0],
                        [0, 0, 128],
                        [128, 0, 128],
                        [0, 128, 128],
                        [128, 128, 128],
                        [64, 0, 0],
                        [192, 0, 0],
                        [64, 128, 0],
                        [192, 128, 0],
                        [64, 0, 128],
                        [192, 0, 128],
                        [64, 128, 128],
                        [192, 128, 128],
                        [0, 64, 0],
                        [128, 64, 0],
                        [0, 192, 0],
                        [128, 192, 0],
                        [0, 64, 128]], dtype='uint8').flatten()
post_processor = crf.DenseCRF(
    iter_max=10,    # 10
    pos_xy_std=3,   # 3
    pos_w=3,        # 3
    bi_xy_std=140,  # 121, 140
    bi_rgb_std=5,   # 5, 5
    bi_w=5,         # 4, 5
)
CEL = nn.CrossEntropyLoss(ignore_index=255).to(device)


def get_model(model_name = "VGG"):
    if(model_name =="VGG"):
        batch_size = 1
        crop_size = 513
        model_path_test = './data/model_last_20000_poly2.pth'
        model = vgg.VGG16_LargeFOV(input_size=crop_size, split='test')
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.load_state_dict(torch.load(model_path_test))
        model.eval()
        model = model.to(device)
    return model

def test_pascal(model_name):
    root_dir_path = './data/VOCdevkit/VOC2012'
    img_dir_path = root_dir_path + '/JPEGImages/'
    gt_dir_path = root_dir_path + '/SegmentationClass/'
    cnn_pred_dir = Path(pred_dir_path+"labels_"+model_name+"_cnn/")
    cnn_pred_dir.mkdir(parents=True, exist_ok=True)
    crf_pred_dir = Path(pred_dir_path+"labels_"+model_name+"_crf/")
    crf_pred_dir.mkdir(parents=True, exist_ok=True)
    model = get_model(model_name)
    batch_size =1
    crop_size = 513
    val_loader = torch.utils.data.DataLoader(
        VOCDataset(split='val', crop_size=crop_size, label_dir_path='SegmentationClassAug', is_scale=False, is_flip=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    times = 0.0
    cnn_mIOU = utils.IOUMetric(num_classes=21)
    crf_mIOU = utils.IOUMetric(num_classes = 21)
    loss_iters, cnn_accuracy_iters,crf_accuracy_iters = [], [],[]
    index =0
    for iter_id, batch in tqdm(enumerate(val_loader)):
        start_time = time.time()
        index = index +1
        image_ids, image, label = batch
        image_id = image_ids[0]
        cnn_img_label,probmap, loss_seg = test_cnn (model,batch)
        
        if TEST_CRF:
            raw_image = cv2.imread(img_dir_path + image_id + '.jpg', cv2.IMREAD_COLOR) # shape = [H, W, 3]
            crf_img_label = test_crf (raw_image,probmap)

        times += time.time() - start_time


        gt_label = Image.open(os.path.join(gt_dir_path, image_id+'.png'))
        w, h = gt_label.size[0], gt_label.size[1]
        gt_img_label = np.array(gt_label, dtype=np.int32)


        cnn_img_label.putpalette(palette)
        cnn_img_label.save(str(cnn_pred_dir)+"/"+image_id + '.png')
        cnn_pred = Image.open(str(cnn_pred_dir)+ "/"+image_id + '.png')
        cnn_pred = cnn_pred.crop((0, 0, w, h))
        cnn_pred = np.array(cnn_pred, dtype=np.int32)
        cnn_mIOU.add_batch(cnn_pred, gt_img_label)

        if TEST_CRF:
            crf_img_label.putpalette(palette)
            crf_img_label.save(str(crf_pred_dir)+ "/"+image_id + '.png')
            crf_pred = Image.open(str(crf_pred_dir)+ "/"+image_id + '.png')
            crf_pred = crf_pred.crop((0, 0, w, h))
            crf_pred = np.array(crf_pred, dtype=np.int32)
            crf_mIOU.add_batch(crf_pred, gt_img_label)
        """
        cnn_label_tensor = torch.LongTensor(np.asarray(cnn_img_label))
        cnn_label_tensor = cnn_label_tensor.to(device)
        crf_label_tensor = torch.LongTensor(np.asarray(cnn_img_label))
        crf_label_tensor = crf_label_tensor.to(device)
        label = label.to(device)
        """
        #cnn_accuracy = float(torch.eq(cnn_label_tensor, label).sum().cpu()) / ( label.shape[1] * label.shape[2])
        #crf_accuracy = float(torch.eq(crf_label_tensor, label).sum().cpu()) / (label.shape[1] * label.shape[2])
        

        #cnn_accuracy_iters.append(float(cnn_accuracy))
        #crf_accuracy_iters.append(float(crf_accuracy))
        loss_iters.append(float(loss_seg.cpu()))

    acc, acc_cls, iou, miou, fwavacc = cnn_mIOU.evaluate()
    print("CNN Metrics")
    print(acc, acc_cls, iou, miou, fwavacc)
    if TEST_CRF:
        acc, acc_cls, iou, miou, fwavacc = crf_mIOU.evaluate()
        print("CRF Metrics")
        print(acc, acc_cls, iou, miou, fwavacc)
        print('dense crf time = %s' % (times / index))
    else:
        print('cnn time = %s' % (times / index))

def test_cnn(model, batch):
    _, image, label = batch
    crop_size =513
    image = image.to(device) #shape =[1,C,H,W]
    labels = utils.resize_labels(label, size=(crop_size, crop_size)).to(device)
    logits = model(image)

    probs = nn.functional.softmax(logits, dim=1) # shape = [batch_size, C, H, W]
    probmap = probs[0].detach().cpu().numpy()

    outputs = torch.argmax(probs, dim=1) # shape = [batch_size, H, W]
    output = np.array(outputs[0].cpu(), dtype=np.uint8)
    img_label = Image.fromarray(output)
    loss_seg = CEL(logits, labels)
    return img_label,probmap,loss_seg

def test_crf(raw_image, probmap):
    h, w = raw_image.shape[:2]
    pad_h = max(513 - h, 0)
    pad_w = max(513 - w, 0)
    pad_kwargs = {
        "top": 0,
        "bottom": pad_h,
        "left": 0,
        "right": pad_w,
        "borderType": cv2.BORDER_CONSTANT,
    }
    raw_image = cv2.copyMakeBorder(raw_image, value=[0, 0, 0], **pad_kwargs)
    raw_image = raw_image.astype(np.uint8)
    prob = post_processor(raw_image, probmap)
    output = np.argmax(prob, axis=0).astype(np.uint8)
    img_label = Image.fromarray(output)
    return img_label



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='VGG', help='test model path')
    parser.add_argument('--model_path_test', default='./exp/model_last_20000_poly2.pth', help='test model path')
    args = parser.parse_args()
    test_pascal(args.model_name)
