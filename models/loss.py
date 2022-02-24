from cv2 import log
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

def centerline_loss_fn(centerlines,logit,label) :
    
    b,_,h,w = label.shape
    counts = []
    predict_index = torch.argmax(logit,dim=1)
    predict_index[label[:,1]==1] = 0
    outside_ratios = []
    #batch len 2

    for i in range(b) :

        predict_dist = torch.stack(torch.where(predict_index[i]>0.5),dim=1).to(dtype=torch.float32)
        #len1 2
        if len(predict_dist) < 6000 and len(predict_dist)>0:
            center_dist = torch.stack(torch.where(centerlines[i]>0.5),dim=1).to(dtype=torch.float32)
            #len2 2
            res = torch.cdist(predict_dist,center_dist)
            #len1 len2
            res = torch.min(res,dim=1)[0]
            filtered_res = res[res>12]
            outside_ratios.append(len(filtered_res)/(len(res)+1)*0.5)
        else :
            outside_ratios.append(0)
    return torch.mean(torch.tensor(outside_ratios,dtype=torch.float32))

class Loss_wrapper(nn.Module):
    
    def __init__(self,args):
        super(Loss_wrapper, self).__init__()
        self.lossfun = None
        self.args = args
        weight = torch.ones((args.num_classes),device='cuda')
        weight[1]*=args.classweight
        if args.loss == 'crossentropy':
            self.lossfun = nn.CrossEntropyLoss(weight=weight)
        elif args.loss == 'dicecrossentropy':
            self.lossfun = nn.CrossEntropyLoss(weight=weight)
            self.dicelossfun = DiceLoss()

    def forward(self, pred, target, weight_centerline=None):
        target = torch.argmax(target,dim=1)
        loss = self.lossfun(pred,target)
        if self.args.loss == 'dicecrossentropy':
            loss+=self.dicelossfun(pred,target,weight_centerline)
        return loss

class Binary_Loss_wrapper(nn.Module):
    def __init__(self,args):
        super(Binary_Loss_wrapper, self).__init__()
        self.num_classes = args.num_classes
        self.loss = None
        if args.loss == "focal":
            self.loss = FocalLoss(logits=True)
        elif args.loss =="dicefocal":
            self.loss = DiceFocalLoss()

    def forward(self, inputs, targets, weight_centerline=None):
        loss = 0
        for i in range(self.num_classes):
            loss+=self.loss(inputs[:,i],targets[:,i],weight_centerline)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1,weight_centerline=None):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        
            
        inputs = F.sigmoid(inputs)
        intersection = (inputs * targets)
        total = (inputs + targets)
        if weight_centerline is not None:
            intersection = intersection*weight_centerline
            total = total * weight_centerline
        
        intersection=intersection.sum()
        total=total.sum()
        
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  

        return dice_loss

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class DiceFocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceFocalLoss, self).__init__()
        self.focal_loss = FocalLoss(logits=False)

    def forward(self, inputs, targets, smooth=1,weight_centerline=None):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        intersection = (inputs * targets)
        total = (inputs + targets)
        if weight_centerline is not None:
            intersection = intersection*weight_centerline
            total = total * weight_centerline
        
        intersection=intersection.sum()
        total=total.sum()
        
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  

        BCE = self.focal_loss(inputs, targets)
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE