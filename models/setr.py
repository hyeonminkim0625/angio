import xdrlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import positionalencoding2d

class convblock(nn.Module):
    """Some Information about convblock"""
    def __init__(self,in_channel,out_channel):
        super(convblock, self).__init__()
        self.head = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(num_features=out_channel),
                                  nn.ReLU())
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
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.decoder = Decoder()
        
        self.cls = nn.Conv2d(256,self.num_classes, 1, padding=0)
        #self.decoder_upscale = nn.Upsample(scale_factor=4, mode='nearest')

        #self.upscale = nn.Upsample(scale_factor=4, mode='nearest')

    def forward(self, x):
        #batch channel h w
        x = self.proj(x) + positionalencoding2d(512,32,32).unsqueeze(0).to('cuda')
        x = x.flatten(2,3).permute(2,0,1)
        x = self.transformer_encoder(x)
        x = x.permute(1,2,0).view(-1,512,32,32)

        #x = self.decoder_upscale(x)
        x = self.decoder(x)
        x = self.cls(x)
        #x = self.upscale(x)
        return {"out" :x}