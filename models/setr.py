import xdrlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
import timm
from utils import positionalencoding2d



class convblock(nn.Module):
    """Some Information about convblock"""
    def __init__(self,in_channel,out_channel):
        super(convblock, self).__init__()
        self.head = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(num_features=out_channel),
                                  nn.ReLU(),
                                  nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(num_features=out_channel),
                                  nn.ReLU()
                                  )
        self.upsample = nn.Upsample(scale_factor = 2, mode='nearest')

    def forward(self, x):
        x = self.head(x)
        x = self.upsample(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        """
        self.head = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(num_features=256),
                                  nn.ReLU(),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(num_features=256),
                                  nn.ReLU(),)
        """
        self.head1 = convblock(512,256)
        self.head2 = convblock(256,256)
        self.head3 = convblock(256,256)
        self.head4 = convblock(256,256)

    def forward(self, x):
        x = self.head1(x)
        x = self.head2(x)
        x = self.head3(x)
        x = self.head4(x)

        return x

class SETR(nn.Module):
    def __init__(self, embed_dim = 512, patch_size = 16):
        super(SETR, self).__init__()
        self.num_classes = 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.layer4_proj = nn.Conv2d(2048, 256, kernel_size=1)
        self.layer3_proj = nn.Conv2d(1024, 128, kernel_size=1)
        self.layer2_proj = nn.Conv2d(512, 128, kernel_size=1)
        self.layer1_proj = nn.Conv2d(256, 128, kernel_size=1)

        model = timm.create_model('resnet50', pretrained=True)
        return_layers = {"layer1": "1", "layer2": "2", "layer3": "3", "layer4": "4"}
        #self.model =  IntermediateLayerGetter(model,return_layers={'patch_embed':'0','norm':'1'})
        self.backbone =  IntermediateLayerGetter(model,return_layers=return_layers)

        encoder_layer = nn.TransformerEncoderLayer(256,8)
        self.transformer_encoder= nn.TransformerEncoder(encoder_layer, num_layers=6)

        decoder_layer = nn.TransformerDecoderLayer(256,8)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=2)

        self.head4 = convblock(256,128)
        self.head3 = convblock(256,128)
        self.head2 = convblock(256,128)
        self.head1 = convblock(256,256)
        #self.head0 = convblock(256,256)
        
        
        
        self.cls = nn.Conv2d(256, self.num_classes, 1, padding = 0)
        #self.decoder_upscale = nn.Upsample(scale_factor=4, mode='nearest')

        self.upscale = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        #batch channel h w
        
        xs = self.backbone(x)
        x4 = self.layer4_proj(xs['4'])
        x3 = self.layer3_proj(xs['3'])
        x2 = self.layer2_proj(xs['2'])
        x1 = self.layer1_proj(xs['1'])

        #b,_,h,w = x.shape
        #x = self.proj(x) + positionalencoding2d(256,h//16,w//16).unsqueeze(0).to('cuda')
        #x = x.flatten(2,3).permute(2,0,1)
        #x = self.transformer_encoder(x)
        #x = x.permute(1,2,0).view(-1,256,h//16,w//16)

        
        b,_,h,w = x4.shape
        x4 = x4 + positionalencoding2d(256,h,w).unsqueeze(0).to('cuda')
        x4 = x4.flatten(2,3).permute(2,0,1)
        x4 = self.transformer_encoder(x4,x)
        x4 = x4.permute(1,2,0).view(-1,256,h,w)
        
        #dfdf
        
        #print(x.shape)
        x4 = self.head4(x4)
        x3 = self.head3(torch.cat((x3,x4),dim=1))
        x2 = self.head2(torch.cat((x2,x3),dim=1))
        x1 = self.head1(torch.cat((x1,x2),dim=1))
        x1 = self.cls(x1)
        x1 = self.upscale(x1)
        #print(x1.shape)

        #x = self.upscale(x)
        return {"out" : x1}


#main.py --batch_size=4 --model=setr --epochs=50 --lr_drop=40 --mask_argmax --opt=adamw --loss=crossentropy --lr=2e-4 --withcoordinate=concat_heatmap --img_size=384 --centerline=False --weight_path=./weight_unet_30/setr_33.pth --mode=val --saveallfig --onlymask --eval --report_hard_sample=30 --output_dir=./result --wandb