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
        #self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.input_proj = nn.Conv2d(1024, 512, kernel_size=1)
        model = timm.create_model('resnet101', pretrained=True)
        return_layers = {"layer1": "1", "layer2": "2", "layer3": "3"}
        #self.model =  IntermediateLayerGetter(model,return_layers={'patch_embed':'0','norm':'1'})
        self.backbone =  IntermediateLayerGetter(model,return_layers=return_layers)

        encoder_layer = nn.TransformerEncoderLayer(512,8)
        self.transformer= nn.TransformerEncoder(encoder_layer, num_layers=6)

        
        self.head1 = convblock(512,256)
        self.head2 = convblock(768,256)
        self.head3 = convblock(512,256)
        self.head4 = convblock(256,256)


        self.cls = nn.Conv2d(256, self.num_classes, 1, padding = 0)
        #self.decoder_upscale = nn.Upsample(scale_factor=4, mode='nearest')

        self.upscale = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        #batch channel h w
        #x = self.proj(x) + positionalencoding2d(512,32,32).unsqueeze(0).to('cuda')
        #x = x.flatten(2,3).permute(2,0,1)
        #x = self.transformer_encoder(x)
        #x = x.permute(1,2,0).view(-1,512,32,32)

        #x = self.decoder_upscale(x)
        
        
        xs = self.backbone(x)

        xs['3'] = self.input_proj(xs['3'])
        b,_,h,w = xs['3'].shape

        xs['3'] = xs['3'] + positionalencoding2d(512,h,w).unsqueeze(0).to('cuda')
        xs['3'] = xs['3'].flatten(2,3).permute(2,0,1)
        xs['3'] = self.transformer(xs['3'])
        xs['3'] = xs['3'].permute(1,2,0).view(-1,512,h,w)

        #print(x.shape)
        xs['3'] = self.head1(xs['3'])
        xs['2'] = self.head2(torch.cat([xs['3'], xs['2']],dim=1))
        xs['1'] = self.head3(torch.cat([xs['2'], xs['1']],dim=1))
        xs['1'] = self.head4(xs['1'])
        xs['1'] = self.cls(xs['1'])
        #x = self.upscale(x)
        return {"out" : xs['1']}


#main.py --batch_size=4 --model=setr --epochs=50 --lr_drop=40 --mask_argmax --opt=adamw --loss=crossentropy --lr=2e-4 --withcoordinate=concat_heatmap --img_size=384 --centerline=False --weight_path=./weight_unet_30/setr_33.pth --mode=val --saveallfig --onlymask --eval --report_hard_sample=30 --output_dir=./result --wandb