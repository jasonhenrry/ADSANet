import os
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np

class test_dataset:
    def __init__(self, image_root, gt_root):
        self.img_list = [os.path.splitext(f)[0] for f in os.listdir(image_root) if f.endswith('.png')]
        self.image_root = image_root
        self.gt_root = gt_root

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.img_list)
        self.index = 0

    def load_test_data(self):
        name = self.image_path[self.index].split('/')[-3]+'-'+self.image_path[self.index].split('/')[-1][:-4]

        image = cv2.imread(self.image_path[self.index])[:,:,::-1].astype(np.float32)
        mask = cv2.imread(self.label_path[self.index], 0).astype(np.float32)

        self.index += 1

        return image, mask


    def load_data(self):

        image = self.rgb_loader(os.path.join(self.image_root,self.img_list[self.index]+ '.png'))
        gt = self.binary_loader(os.path.join(self.gt_root,self.img_list[self.index] + '.png'))
        name = self.img_list[self.index].split('/')[-1] + '.png'

        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

