import torch
import torch.nn as nn
import torch.nn.functional as F
from model.res2net import res2net50_v1b_26w_4s
import torchvision


class ADSANet(nn.Module):
    # res2net based encoder decoder
    def __init__(self):
        super(ADSANet, self).__init__()
        
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        self.x5_dem_1 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_dem_1 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x5_x4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.x5_x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.level4concatconv = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level3concatconv = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level2concatconv = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level1concatconv = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.level3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.output4_c = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3_c = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2_c = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.output1_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))
        
        self.predict = nn.Conv2d(64*3, 1, kernel_size=1, stride=1, padding=0)

        self.foutput_c =  nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        input = x

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x1 = self.resnet.maxpool(x) 

        x2 = self.resnet.layer1(x1) 
        x3 = self.resnet.layer2(x2) 
        x4 = self.resnet.layer3(x3) 
        x5 = self.resnet.layer4(x4) 

        x5_dem_1 = self.x5_dem_1(x5)
        x4_dem_1 = self.x4_dem_1(x4)
        x3_dem_1 = self.x3_dem_1(x3)
        x2_dem_1 = self.x2_dem_1(x2)

        out5 = F.interpolate(x5_dem_1, size=x3.size()[2:], mode='bilinear', align_corners=True)
        out4 = F.interpolate(x4_dem_1, size=x3.size()[2:], mode='bilinear', align_corners=True)
        pred = torch.cat([out5, out4*out5, x3_dem_1*out4*out5], dim=1)
        
        pred = self.predict(pred)

        x5_4 = self.x5_x4(abs(F.upsample(x5_dem_1,size=x4.size()[2:], mode='bilinear')-x4_dem_1))
        level4 = torch.cat((x5_4, (F.upsample(x5_dem_1,size=x5_4.size()[2:], mode='bilinear'))), 1)
        output4 = self.output4_c(torch.cat((self.level4concatconv(level4), x4_dem_1), 1))

        x4_3 = self.x4_x3(abs(F.upsample(output4,size=x3.size()[2:], mode='bilinear')-x3_dem_1))
        level3 = self.level3(x4_3)
        x5_4_3 = self.x5_x4_x3(abs(F.upsample(x5_4, size=x4_3.size()[2:], mode='bilinear') - level3))
        output3 = self.output3_c(torch.cat((self.level3concatconv(torch.cat((x5_4_3, level3, F.upsample(output4,size=level3.size()[2:], mode='bilinear')), 1)), x3_dem_1), 1))

        x3_2 = self.x3_x2(abs(F.upsample(output3,size=x2.size()[2:], mode='bilinear')-x2_dem_1))
        level2 = self.level2(x3_2)
        x4_3_2 = self.x4_x3_x2(abs(F.upsample(level3, size=x3_2.size()[2:], mode='bilinear') - level2))
        output2 = self.output2_c(torch.cat((self.level2concatconv(torch.cat((x4_3_2, level2, F.upsample(output3,size=level2.size()[2:], mode='bilinear')), 1)), x2_dem_1), 1))

        x2_1 = self.x2_x1(abs(F.upsample(output2,size=x1.size()[2:], mode='bilinear')-x1))
        level1 = self.level1(x2_1)
        x3_2_1 = self.x3_x2_x1(abs(F.upsample(level2, size=x2_1.size()[2:], mode='bilinear') - level1))
        
        level1 = torch.cat((x3_2_1, level1, F.upsample(output2,size=level1.size()[2:], mode='bilinear')), 1)
        output1 = self.output1_1(self.level1concatconv(level1))
        output1 = self.output1(output1)

        fpred = torch.cat([F.upsample(pred, size=level1.size()[2:], mode='bilinear'), output1], dim=1)
        fpred = self.foutput_c(fpred)

        output = F.upsample(fpred, size=input.size()[2:], mode='bilinear')

        if self.training:
            return output
        return output

if __name__ == '__main__':
    ras = ADSANet().cuda()
    
    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(ras, (3, 352, 352), as_strings=True, print_per_layer_stat=True)
    print('macs: ', macs, 'params: ', params)  # flops = macs*2
