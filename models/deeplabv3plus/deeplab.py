import torch
import torch.nn as nn
import torch.nn.functional as F

from models.deeplabv3plus.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.deeplabv3plus.aspp import build_aspp
from models.deeplabv3plus.decoder import build_decoder
from models.deeplabv3plus.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder1 = build_decoder(num_classes, backbone, BatchNorm, 128)
        self.decoder2 = build_decoder(num_classes, backbone, BatchNorm, 64)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        #x, low_level_feat = self.backbone(input)
        low_level_feat_,low_level_feat,x,x_ = self.backbone(input)
        
        x_ = F.interpolate(x_, size=x.size()[2:], mode='nearest')
        x = self.aspp(torch.cat((x,x_),dim=1))
        
        x = self.decoder1(x, low_level_feat)
        x = self.decoder2(x, low_level_feat_)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return {'out':x}

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
    model = DeepLab(backbone='xception', output_stride=16,num_classes=3,sync_bn=False)
    model.eval()
    input = torch.rand(1, 3, 256, 256)
    output = model(input)
    print(output.size())


