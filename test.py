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
import itertools
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
palette[:3*21] = np.array([[0, 0, 0],[128, 0, 0],[0, 128, 0],[128, 128, 0],[0, 0, 128],[128, 0, 128],[0, 128, 128],[128, 128, 128],[64, 0, 0],[192, 0, 0],
    [64, 128, 0],[192, 128, 0],[64, 0, 128],[192, 0, 128],[64, 128, 128],[192, 128, 128],[0, 64, 0],[128, 64, 0],[0, 192, 0],[128, 192, 0],[0, 64, 128]], 
    dtype='uint8').flatten()
post_processor = crf.DenseCRF(
    iter_max=10,    
    pos_xy_std=3,   
    pos_w=3,        
    bi_xy_std=140,  
    bi_rgb_std=5,
    bi_w=5,         
)

post_processor_step = crf.DenseCRF_step(
    iter_max=10,    
    pos_xy_std=3,   
    pos_w=3,        
    bi_xy_std=140,  
    bi_rgb_std=5,
    bi_w=5,         
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

def test_pascal(model_name, use_crf):
    TEST_CRF = False
    if use_crf:
        TEST_CRF = True
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
        cnn_img_label,probmap = test_cnn (model,batch)
        
        if TEST_CRF:
            raw_image = cv2.imread(img_dir_path + image_id + '.jpg', cv2.IMREAD_COLOR) # shape = [H, W, 3]
            crf_img_label = test_crf (raw_image,probmap,post_processor)

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
    logits = model(image)
    probs = nn.functional.softmax(logits, dim=1) # shape = [batch_size, C, H, W]
    probmap = probs[0].detach().cpu().numpy()

    outputs = torch.argmax(probs, dim=1) # shape = [batch_size, H, W]
    output = np.array(outputs[0].cpu(), dtype=np.uint8)
    img_label = Image.fromarray(output)
    return img_label,probmap

def test_crf(raw_image, probmap,post_processor):
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

def test_crf_step(raw_image, probmap,post_processor):
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
    Q_list, kl_loss = post_processor(raw_image, probmap)
    pred_list=[]
    for Q in Q_list:
        output = np.argmax(Q, axis=0).astype(np.uint8)
        img_label = Image.fromarray(output)
        pred_list.append(img_label)
    
    return pred_list,kl_loss




class DeepLab_v1():
    def __init__(self, num_classes, gpu_id=0, weight_file=None):
        self.num_classes = num_classes
        
        self.gpu = gpu_id

        torch.cuda.set_device(self.gpu)
        crop_size =513
        batch_size =1
        self.model = get_model()
        self.root_dir_path = './data/VOCdevkit/VOC2012'
        self.img_dir_path = self.root_dir_path + '/JPEGImages/'
        self.gt_dir_path = self.root_dir_path + '/SegmentationClass/'
        self.dataset  = VOCDataset(split='val', crop_size=crop_size, label_dir_path='SegmentationClassAug', is_scale=False, is_flip=False)
        self.val_loader = torch.utils.data.DataLoader(self.dataset,batch_size=batch_size,shuffle=False,num_workers=2,drop_last=False)
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).cuda(self.gpu, non_blocking=True)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).cuda(self.gpu, non_blocking=True)
        
        self.eps = 1e-10
        self.best_mIoU = 0.
        self.iter_max = 10
        self.bi_w = 7
        self.bi_xy_std = 50
        self.bi_rgb_std = 4
        self.pos_w = 3
        self.pos_xy_std = 3

    def grid_search(self, iter_max, bi_ws, bi_xy_stds, bi_rgb_stds, pos_ws, pos_xy_stds):
        self.model.eval()
        mIoU_list =[]
        metrics_list =[]
        with torch.no_grad():
            kl_array=[]
            for bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std in itertools.product(bi_ws, bi_xy_stds, bi_rgb_stds, pos_ws, pos_xy_stds):
                crf_mIOU = utils.IOUMetric(num_classes = 21)
                
                
                for iter_id, batch in tqdm(enumerate(self.val_loader)):
                    if iter_id == 100: break
                    image_ids, image, label = batch
                    image_id = image_ids[0]
                    gt_label = Image.open(os.path.join(self.gt_dir_path, image_id+'.png'))
                    w, h = gt_label.size[0], gt_label.size[1]
                    gt_img_label = np.array(gt_label, dtype=np.int32)
                    Q_list , kl = self.inference_step(image_id, "CRF",iter_max,bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std )
                    kl_array.append(kl)
                    crf_pred  =Q_list[-1]
                    crf_pred = crf_pred.crop((0, 0, w, h))
                    crf_pred = np.array(crf_pred, dtype=np.int32)
                    crf_mIOU.add_batch(crf_pred, gt_img_label)

                metrics = crf_mIOU.evaluate()
                mIoU = metrics[3]
                mIoU_list.append(mIoU)
                metrics_list.append(metrics)
                
                state = ('bi_w : {}, bi_xy_std : {}, bi_rgb_std : {}, pos_w : {}, pos_xy_std : {}  '
                         'mIoU : {:.4f}').format(bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std, 100 * mIoU)
                
                if mIoU > self.best_mIoU:
                    print()
                    print('*' * 35, 'Best mIoU Updated', '*' * 35)
                    print(state)
                    self.best_mIoU = mIoU
                return metrics_list, mIoU_list, np.mean(kl_array)
                    
    def inference(self, image_path, model_type, iter_max, bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std):
        self.model.eval()
        TEST_CRF = False
        with torch.no_grad():
            if(model_type == "CRF"):
              TEST_CRF = True
            image_id = os.path.split(image_path)[1]
            image_id = image_id.split('.')[0]
            image_id, image, label = self.dataset.__getitem__(image_path,image_id)
            
            image_tensor = torch.FloatTensor(image)
            image_tensor = image_tensor.unsqueeze(0)
            label  = torch.LongTensor(label)
            label = label.unsqueeze(0)
            label = label.unsqueeze(0)
           
            cnn_img_label,probmap =  test_cnn(self.model, (image_id,image_tensor,label))
            cnn_img_label.putpalette(palette)
            #image = np.transpose(image,(1,2,0))
            if TEST_CRF:
                raw_image = cv2.imread(self.img_dir_path + image_id + '.jpg', cv2.IMREAD_COLOR) # shape = [H, W, 3]
                crf_img_label = test_crf (raw_image,probmap,post_processor)
                crf_img_label.putpalette(palette)                
                return crf_img_label, crf_img_label
            return cnn_img_label, probmap

    def inference_step(self, image_path, model_type, iter_max, bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std):
        self.model.eval()
        with torch.no_grad():
            image_id = os.path.split(image_path)[1]
            image_id = image_id.split('.')[0]
            image_id, image, label = self.dataset.__getitem__(image_path,image_id)
            
            image_tensor = torch.FloatTensor(image)
            image_tensor = image_tensor.unsqueeze(0)
            label  = torch.LongTensor(label)
            label = label.unsqueeze(0)
            label = label.unsqueeze(0)
            image = np.transpose(image,(1,2,0))
            crf_processor = crf.DenseCRF_step(iter_max, bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std)
            cnn_img_label,probmap =  test_cnn(self.model, (image_id,image_tensor,label))
            cnn_img_label.putpalette(palette)
            #image = np.transpose(image,(1,2,0))
            raw_image = cv2.imread(self.img_dir_path + image_id + '.jpg', cv2.IMREAD_COLOR) # shape = [H, W, 3]
            Q_list, kl_loss = test_crf_step (raw_image,probmap,crf_processor)
            for Q in Q_list:
                Q.putpalette(palette)
            return Q_list, kl_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_crf',default=False)
    parser.add_argument('--model_name', default='VGG', help='test model path')
    parser.add_argument('--model_path_test', default='./exp/model_last_20000_poly2.pth', help='test model path')
    args = parser.parse_args()
    test_pascal(args.model_name, args.use_crf)
