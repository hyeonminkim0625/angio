import torch
import torch.nn as nn
import torch.nn.functional as F

from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from aspp import build_aspp
from decoder import build_decoder
from backbone import build_backbone
from position_encoding import PositionEmbeddingSine

class DeepLabv3_Transformer(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLabv3_Transformer, self).__init__()
        if backbone == 'drn':
            output_stride = 8
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

        self.ps = PositionEmbeddingSine(num_frames=5)
        self.transformer = nn.Transformer(d_model=384)

    def forward(self, input):
        input={'anchor' : torch.randn((2,3,256,256)),'support' : [torch.randn((2,3,256,256)) for _ in range(4)]}

        x_, low_level_feat = self.backbone(input['anchor'])
        x_ = self.aspp(x_)
        temp = [x_]

        for support in input['support']:
            x, _ = self.backbone(support)
            x = self.aspp(x)
            temp.append(x)
        
        temp = torch.stack(temp,dim=1)
        temp += self.ps(temp)
        temp = torch.flatten(temp.permute(0,2,1,3,4),start_dim=2,end_dim=-1)

        x_=x_.unsqueeze(1)
        x_ += self.ps(x_)
        x_ = torch.flatten(x_.permute(0,2,1,3,4),start_dim=2,end_dim=-1)

        x_ = self.transformer(temp.permute(2,0,1),x_.permute(2,0,1))


        x_ = x_.permute(1,2,0).view(-1,384,16,16)
        x_ = self.decoder(x_, low_level_feat)
        x_ = F.interpolate(x_, size=input['anchor'].size()[2:], mode='bilinear', align_corners=True)

        return x

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
    model = DeepLabv3_Transformer(backbone='xception', output_stride=16,num_classes=3,sync_bn=False)
    model.eval()
    input = torch.rand(1, 3, 256, 256)
    output = model(input)
