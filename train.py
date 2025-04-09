#!/usr/bin/python3
#coding=utf-8
import os
import datetime
os.environ['CUDA_VISIBLE_DEVICE']='0'
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda import amp
from utils import dataset_medical
from model.ADSANet import ADSANet

torch.cuda.set_device(0)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=50):
    decay = decay_rate ** (epoch / decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

use_fp16 = True
scaler = amp.GradScaler(enabled=use_fp16)

dataset_medical.epochnum = 1
def train(Dataset, Network, savepath):
    ## dataset
    train_path = './data/TrainDataset'

    cfg = Dataset.Config(datapath=train_path, savepath=savepath,\
    mode='train', batch=16, lr=0.05, momen=0.9, decay=5e-4, epoch=50)

    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=4)
    if not os.path.exists(cfg.savepath):
        os.makedirs(cfg.savepath)
        
    net = Network()
    net.train(True)
    net.cuda()
    torch.backends.cudnn.enabled = False
    base, head = [], []
    # cnt=0
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'resnet' in name:
            base.append(param)
        else:
            head.append(param)

    global_step    = 0
    optimizer = torch.optim.AdamW([{'params':base}, {'params':head}], lr=5*1e-4, weight_decay=1e-4)
    
    for epoch in range(0, cfg.epoch):

        dataset_medical.epochnum = epoch

        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step, (image, mask) in enumerate(loader):
            image, mask = image.cuda().float(), mask.cuda().float()
            with amp.autocast(enabled=use_fp16):
                output = net(image)
                loss1u = structure_loss(output, mask)
                loss = loss1u
            optimizer.zero_grad()

            loss.backward()
            clip_gradient(optimizer, 0.5)
            optimizer.step()

            global_step += 1
            if step %10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss1u=%.6f'%(datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss1u.item()))

        if epoch>47:
        # if epoch>-1:
            torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))
            print(cfg.savepath+'/model-'+str(epoch+1))
            print('do saving')


if __name__=='__main__':
    path='./saved_model'
    train(dataset_medical, ADSANet, path)
