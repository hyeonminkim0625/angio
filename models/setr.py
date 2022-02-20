import xdrlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import positionalencoding2d

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.head = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(num_features=256),
                                  nn.ReLU(),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(num_features=256),
                                  nn.ReLU(),)

    def forward(self, x):
        x = self.head(x)

        return x

class SETR(nn.Module):
    def __init__(self, embed_dim = 256, patch_size = 16):
        super(SETR, self).__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm_layer = nn.LayerNorm(256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.decoder = Decoder()
        self.num_classes = 2
        self.cls = nn.Conv2d(256,self.num_classes, 1, padding=0)

    def forward(self, x):
        #batch channel h w
        x = self.proj(x) + positionalencoding2d(256,32,32).unsqueeze(0).to('cuda')
        x = x.flatten(2,3).permute(2,0,1)
        x = self.transformer_encoder(x)
        x = x.permute(1,2,0).view(-1,256,32,32)
        x = self.decoder(x)
        x = self.cls(x)
        return {"out" :x}