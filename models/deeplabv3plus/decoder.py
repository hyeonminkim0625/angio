import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deeplabv3plus.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class Decoder_revised(nn.Module):
    """Some Information about Decoder_revised"""
    def __init__(self,in_channel,out_channel,scale_factor):
        super(Decoder_revised, self).__init__()
        self.head = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(num_features=out_channel),
                                  nn.ReLU(),
                                  nn.Dropout(0.1),
                                  nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(num_features=out_channel),
                                  nn.ReLU(),
                                  nn.Dropout(0.1),
                                  )
        self.upsample = nn.Upsample(scale_factor = scale_factor, mode='bilinear', align_corners=True)

    def forward(self, x,low_feature):
        x = self.upsample(x)
        x = torch.cat((x,low_feature),dim=1)
        x = self.head(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm,low_level_inplanes):
        super(Decoder, self).__init__()
        

        self.conv1 = nn.Conv2d(low_level_inplanes, 64, 1, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(320, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm,low_level_inplanes):
    return Decoder(num_classes, backbone, BatchNorm,low_level_inplanes)