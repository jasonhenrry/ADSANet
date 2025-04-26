#!/usr/bin/python3
#coding=utf-8

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFile

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image, mask):
        image = (image - self.mean)/self.std
        mask /= 255
        return image, mask

class RandomCrop(object):
    def __call__(self, image, mask):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        return image[p0:p1,p2:p3], mask[p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2)==0:
            return image[:, ::-1], mask[:, ::-1]
        else:
            return image, mask

class RandomVFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2)==0:
            return image[::-1,:], mask[::-1, :]
        else:
            return image, mask

class RandomRotate(object):
    def __call__(self, image, mask):
        degree = 10
        rows, cols, channels = image.shape
        random_rotate = random.random() * 2 * degree - degree
        rotate = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), random_rotate, 1)
        image = cv2.warpAffine(image, rotate, (cols, rows))
        mask = cv2.warpAffine(mask, rotate, (cols, rows))

        return image,mask

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask

class ToTensor(object):
    def __call__(self, image, mask):
        image = torch.from_numpy(image)
        # image = image.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        return image, mask

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None

global epochnum

class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.randomVflip = RandomVFlip()
        self.randomrotate = RandomRotate()
        self.resize     = Resize(352, 352)
        self.totensor   = ToTensor()

        self.root = cfg.datapath

        img_path = os.path.join(self.root, 'image')
        gt_path = os.path.join(self.root, 'mask')
        self.samples = [os.path.splitext(f)[0]
                    for f in os.listdir(img_path) if f.endswith('.png') or f.endswith('.bmp')]

        self.color1, self.color2 = [], []
        for name in self.samples:
            if name[:].isdigit():
                self.color1.append(name)
            else:
                self.color2.append(name)
    
    def __getitem__(self, idx):
        global epochnum
        name  = self.samples[idx]

        # Color exchange with region suppressing
        image = cv2.imread(self.root+'/image/'+name+'.png')
        imaget = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imaget = np.float32(imaget)
        mask  = cv2.imread(self.root+'/mask/' +name+'.png', 0).astype(np.float32)

        imgray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        maskblack1 = np.uint8(imgray1<8)

        kernel = np.ones((3,3), np.uint8)
        maskblack1 = cv2.erode(maskblack1, kernel, iterations=1)
        maskblack1 = cv2.dilate(maskblack1, kernel, iterations=1)
        
        maskspec1 = np.uint8(imgray1>220)
        mask1 = np.uint8((1-(maskblack1+maskspec1)>0)*255)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        (mean, std) = cv2.meanStdDev(image, mask=mask1)
        mean = mean.reshape(1,1,3)
        std = std.reshape(1,1,3)

        rnum = np.random.randint(0, 1500)
        name2  = self.color1[(idx+rnum)%len(self.color1)] if np.random.rand()<0.7 else self.color2[(idx+rnum)%len(self.color2)]
        image2 = cv2.imread(self.root+'/image/'+name2+'.png')
        imgray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        maskblack2 = np.uint8(imgray2<8)

        maskblack2 = cv2.erode(maskblack2, kernel, iterations=1)
        maskblack2 = cv2.dilate(maskblack2, kernel, iterations=1)
        
        maskspec2 = np.uint8(imgray2>220)
        mask2 = np.uint8((1-(maskblack2+maskspec2)>0)*255)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)
        (mean2, std2) = cv2.meanStdDev(image2, mask=mask2)
        mean2 = mean2.reshape(1,1,3)
        std2 = std2.reshape(1,1,3)
        
        image_ce = np.float32((image-mean)/(std+1e-8)*std2+mean2)
        maskc = np.expand_dims((1-maskblack1)*255, axis=2)
        maskc = np.concatenate((maskc, maskc, maskc), axis=-1)

        cond = image_ce>255
        image_ce[cond] = 255
        cond = image_ce<0
        image_ce[cond] = 0

        image_ce = cv2.convertScaleAbs(image_ce)

        image_ce = np.uint8(image_ce*(maskc/255) + image*((255-maskc)/255))
        image = cv2.cvtColor(image_ce, cv2.COLOR_LAB2RGB)
        image = np.asarray(image, np.float32)
        # Color exchange with region suppressing
       
        shape = mask.shape

        if self.cfg.mode=='train':
            image, mask = self.normalize(image, mask)
            image, mask = self.resize(image, mask)
            image, mask = self.randomcrop(image, mask)
            image, mask = self.randomflip(image, mask)
            image, mask = self.randomVflip(image, mask)
            image, mask = self.randomrotate(image, mask)
            return image, mask
        else:
            image, mask = self.normalize(imaget, mask)
            image, mask = self.resize(image, mask)
            image, mask = self.totensor(image, mask)
            return image, mask, shape, name

    def collate(self, batch):
        size = 352
        image, mask = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        return image, mask

    def __len__(self):
        return len(self.samples)


