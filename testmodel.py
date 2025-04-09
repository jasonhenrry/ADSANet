from statistics import median
import numpy as np
from numpy.core.fromnumeric import squeeze
import torch
import cv2
from utils.test_data import test_dataset
from utils.metric import cal_biou, cal_mae,cal_fm, cal_prec, cal_sens,cal_sm,cal_em, cal_spec,cal_wfm, cal_dice, cal_iou,cal_ber,cal_acc
from model.ADSANet import ADSANet
import torch.nn.functional as F
import os
from utils.misc import crf_refine

test_datasets = ['CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir', 'CVC-300', 'CVC-ClinicDB']

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image, mask):
        image = (image - self.mean)/self.std
        mask /= 255
        return image, mask

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask

class ToTensor(object):
    def __call__(self, image):
        image = torch.from_numpy(image)
        return image

path=(['./saved_model/model-50']) 


dir = '/data2/data/hkl/Data/Kvasir-CVC612-MSNet/TestDataset/'
dcnt = 0


for dataset in test_datasets:
    if brkf==True:
        break

    save_path = './results/nCrf/{}/'.format(dataset)
    os.makedirs(save_path, exist_ok=True)
    # print(dcnt)
    dataset_path =  dir+test_datasets[dcnt]+'/masks'
    dataset_path_pre = dir+test_datasets[dcnt]+'/images'
    dcnt = dcnt+1
    sal_root = dataset_path_pre  +'/'
    gt_root = dataset_path  +'/'
    test_loader = test_dataset(sal_root, gt_root)
    mae,fm,sm,em,wfm, m_dice, m_piou,m_biou,ber,acc, spec, sens, prec= cal_mae(),cal_fm(test_loader.size),cal_sm(),cal_em(),cal_wfm(), cal_dice(), cal_iou(),cal_biou(),cal_ber(),cal_acc(), cal_spec(),cal_sens(),cal_prec()

    mean = np.array([[[124.55, 118.90, 102.94]]])
    std  = np.array([[[ 56.77,  55.97,  57.50]]])
    normalize = Normalize(mean, std)
    resize = Resize(352, 352)
    totensor = ToTensor()

    model = path
    net = ADSANet()
    net.cuda()
    net.load_state_dict(torch.load(model), False)
    net.eval()
    for i in range(test_loader.size):
        img, mask, name = test_loader.load_data()
        img = np.array(img, dtype='float32')
        mask = np.array(mask, dtype='float32')
        shape = mask.shape
        imgR, maskT = resize(img, mask)
        imgT, maskT = normalize(imgR, maskT)
        
        imgT = totensor(imgT)

        imgT = imgT.permute(2, 0, 1)
        imgT = imgT.unsqueeze(0)
        imgT = imgT.to(torch.float32)
        imgT = imgT.cuda().float()

        output = net(imgT)
        sal = F.sigmoid(output)

        sal = sal.cpu()

        sal = sal.squeeze()
        sal = sal.squeeze()
        sal = sal.detach().numpy()
        gt = maskT

        if sal.size != gt.size:
            x, y = gt.size
            sal = sal.resize((x, y))
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        gt[gt > 0.5] = 1
        gt[gt != 1] = 0
        res = sal
        res = np.array(res)

        res_s = cv2.resize(res, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        fname = save_path + name
        cv2.imwrite(fname, np.uint8(res_s*255))
        
        ##########crf
        res = np.asarray(res*255, np.uint8)
        imgR = np.asarray(imgR*255, np.uint8)
        res = crf_refine(imgR, res)
        res = np.asarray(res, np.float32)
        res = res/255
        ##########crf

        mae.update(res, gt)
        sm.update(res,gt)
        em.update(res,gt)
        wfm.update(res,gt)
        m_dice.update(res,gt)
        m_piou.update(res,gt)

    MAE = mae.show()
    sm = sm.show()
    em = em.show()
    wfm = wfm.show()
    m_dice = m_dice.show()
    m_piou = m_piou.show()
    

    print('dataset: {} MAE: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f} M_dice: {:.4f} M_iou: {:.4f}'.format(dataset, MAE, wfm, sm, em, m_dice, m_piou))
    