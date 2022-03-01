import torch
import torch.nn as nn
import torch.nn.functional as F

from models.deeplabv3plus.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.deeplabv3plus.aspp import build_aspp
from models.deeplabv3plus.decoder import build_decoder, Decoder_revised
from models.deeplabv3plus.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone,args):
        super(DeepLab, self).__init__()
        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, 16, BatchNorm)
        self.aspp = build_aspp(backbone, 16, BatchNorm,args)

        self.decoder1 = Decoder_revised(384+256,256,2,args)
        self.decoder2 = Decoder_revised(256+192,256,2,args)
        self.cls = nn.Conv2d(256, 2, 1, padding = 0)

    def forward(self, input):
        x1,x2,x3,x4 = self.backbone(input)
        
        x4 = F.interpolate(x4, size=x3.size()[2:], mode='bilinear', align_corners=True)

        x3 = self.aspp(torch.cat((x3,x4),dim=1))
        
        x2 = self.decoder1(x3, x2)
        x1 = self.decoder2(x2, x1)

        x1 = self.cls(x1)
        x1 = F.interpolate(x1, size=input.size()[2:], mode='bilinear', align_corners=True)

        return {'out':x1}

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    model = DeepLab(backbone='convnext', output_stride=8, num_classes=2,sync_bn=False, freeze_bn=False)
    model.eval()
    input = torch.rand(1, 3, 256, 256)
    output = model(input)
    print(output.size())