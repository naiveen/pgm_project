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

import itertools
from PIL import Image

import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

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


class DenseCRF_step(object):
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

        Q_list= []
        

        kl_loss =[]
        Q, tmp1, tmp2 = d.startInference()
        Q_ = np.array(Q).reshape((C, H, W))
        Q_list.append(Q_)
        for i in range(self.iter_max):
            kl_loss.append( d.klDivergence(Q))
            d.stepInference(Q, tmp1, tmp2)
            Q_ = np.array(Q).reshape((C, H, W))
            Q_list.append(Q_)


        return Q_list, kl_loss



