#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   09 January 2019





import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms.functional as F

import itertools, utils
from PIL import Image

import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils


class DeepLab_v1():
    def __init__(self, num_classes, gpu_id=0, weight_file=None):
        self.num_classes = num_classes
        
        self.gpu = gpu_id

        torch.cuda.set_device(self.gpu)

        self.model = VGG16_LargeFOV(self.num_classes, init_weights=False).cuda(self.gpu)
        self.model.load_state_dict(torch.load(weight_file))
        
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).cuda(self.gpu, non_blocking=True)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).cuda(self.gpu, non_blocking=True)
        
        self.eps = 1e-10
        self.best_mIoU = 0.

    def grid_search(self, data, iter_max, bi_ws, bi_xy_stds, bi_rgb_stds, pos_ws, pos_xy_stds):

        self.model.eval()
        with torch.no_grad():
            for bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std in itertools.product(bi_ws, bi_xy_stds, bi_rgb_stds, pos_ws, pos_xy_stds):
                
                tps = torch.zeros(self.num_classes).cuda(self.gpu, non_blocking=True)
                fps = torch.zeros(self.num_classes).cuda(self.gpu, non_blocking=True)
                fns = torch.zeros(self.num_classes).cuda(self.gpu, non_blocking=True)
                
                crf = DenseCRF(iter_max, bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std)
                
                for i, (image, y) in enumerate(data):
                    
                    if i == 100: break
                        
                    n, c, h, w = y.shape
                    y = y.view(n, h, w).type(torch.LongTensor)
                    
                    X = image
                    X, y = X.cuda(self.gpu, non_blocking=True), y.cuda(self.gpu, non_blocking=True)
                    X = X.float().div(255)
                    
                    X = X.sub_(self.mean).div_(self.std)
                    
                    output = self.model(X)
                    output = F.resize(output, (h, w), Image.BILINEAR)
                    output = nn.Softmax2d()(output)

                    for j in range(n):
                        predict = crf(image[j], output[j])
                        predict = torch.from_numpy(predict).float().cuda(self.gpu, non_blocking=True)
                        predict = torch.argmax(predict, dim=0)
                        
                        filter_255 = y!=255
                        
                        for i in range(self.num_classes):
                            positive_i = predict==i
                            true_i = y==i
                            tps[i] += torch.sum(positive_i & true_i)
                            fps[i] += torch.sum(positive_i & ~true_i & filter_255)
                            fns[i] += torch.sum(~positive_i & true_i)
                
                mIoU = torch.sum(tps / (self.eps + tps + fps + fns)) / self.num_classes
                
                state = ('bi_w : {}, bi_xy_std : {}, bi_rgb_std : {}, pos_w : {}, pos_xy_std : {}  '
                         'mIoU : {:.4f}').format(bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std, 100 * mIoU)
                
                if mIoU > self.best_mIoU:
                    print()
                    print('*' * 35, 'Best mIoU Updated', '*' * 35)
                    print(state)
                    self.best_mIoU = mIoU
                else:
                    print(state)
                    
    def inference(self, image_dir, iter_max, bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std):
        self.model.eval()
        with torch.no_grad():
            image = Image.open(image_dir).convert('RGB')
            
            image_tensor = torch.as_tensor(np.asarray(image))
            image_tensor = image_tensor.view(image.size[1], image.size[0], len(image.getbands()))
            image_tensor = image_tensor.permute((2, 0, 1))
            
            c, h, w = image_tensor.shape
            image_norm_tensor = image_tensor[None, ...].float().div(255).cuda(self.gpu, non_blocking=True)
            
            image_norm_tensor = image_norm_tensor.sub_(self.mean).div_(self.std)
            
            output = self.model(image_norm_tensor)
            output = F.resize(output, (h, w), Image.BILINEAR)
            output = nn.Softmax2d()(output)
            output = output[0]
            
            crf = DenseCRF(iter_max, bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std)
            
            predict = crf(image_tensor, output)
            predict = np.argmax(predict, axis=0)
            return predict


class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )

        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q


