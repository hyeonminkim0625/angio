import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from models.convlstm import ConvLSTM
from models.unet_plusplus import Nested_UNet
from models.deeplabv3plus.deeplab import DeepLab
"""
https://github.com/niecongchong/HRNet-keras-semantic-segmentation/blob/master/model/seg_hrnet.py
"""
class Consistency_model_wrapper(nn.Module):
    """Some Information about Consistdddddency_model_wrapper"""
    def __init__(self,model):
        super(Consistency_model_wrapper, self).__init__()
        self.crop = torchvision.transforms.RandomCrop(256)
        self.model = model

    def forward(self, x):
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(x, output_size=(224, 224))
        random_cropped_img = TF.resized_crop(x, i, j, h, w, 256)
        
        return [self.model(x), self.model(random_cropped_img),(i,j,h,w)]

class BaseLine_wrapper(nn.Module):
    """
    deeplabv3_resnet_50
    """
    def __init__(self,args):
        super(BaseLine_wrapper, self).__init__()
        self.model = None
        self._model = args.model
        self.args = args
        _num_classes = args.num_classes
        channel = 3
        _num_classes = args.num_classes
        if self._model == "deeplab":
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False,num_classes=_num_classes)
        elif self._model == "unet":
            self.model = UNet(channel,_num_classes)
        elif self._model == "fcn":
            self.model = torchvision.models.segmentation.fcn_resnet50(pretrained=False,num_classes=_num_classes)
        elif self._model == "unetpp":
            self.model = Nested_UNet(_num_classes,3,deep_supervision=True)
        elif args.model == 'deeplabv3plus':
            self.model = DeepLab(backbone='resnet', output_stride=8, num_classes=_num_classes,sync_bn=False, freeze_bn=False)
    def forward(self, x):
        x = self.model(x)['out']

        return x

class LSTM_wrapper(nn.Module):
    """
    deeplabv3_resnet_50
    """
    def __init__(self,args):
        super(LSTM_wrapper, self).__init__()
        self.model = None
        self._model = args.model
        channel = 3
        if args.frame!=0:
            channel = args.frame*2 + 1
        _num_classes = args.num_classes
        if self._model == "deeplab":
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False,num_classes=_num_classes)
        elif self._model == "unet":
            self.model = UNet(channel,_num_classes)
        elif self._model == "fcn":
            self.model = torchvision.models.segmentation.fcn_resnet50(pretrained=False,num_classes=_num_classes)
        
        self.convlstm = ConvLSTM(input_dim=3,hidden_dim=3,kernel_size=(3,3),num_layers=1,batch_first=False)

    def forward(self, x):
        temp = []
        for support in x['support']:
            temp.append(self.model(support)['out'])
        temp.append(self.model(x['anchor'])['out'])
        
        layer_output_list, _ =self.convlstm(torch.stack(temp,dim=0))#T, B, C, H, W -> B, T, C, H, W
        return layer_output_list[:,-1]

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return {"out" : self.conv(dec1)}

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
