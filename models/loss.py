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

    for i in range(b) :
        mins = []
        
        def get_mins(x,y) :
            distances = []
            for c1,c2 in zip(torch.where(centerlines[i]>0.5)[0],torch.where(centerlines[i]>0.5)[1]) :
                distances.append(((x-c1)**2+(y-c2)**2)**0.5)

            return min(distances)
        
        
        xs = torch.where(predict_index[i]>0.5)[0]
        ys = torch.where(predict_index[i]>0.5)[1]
        
        if len(xs) < 6000 :
            mins = list(map(lambda x,y: get_mins(x,y), xs,ys))
            outside_ratios.append(len(mins[mins>12])/(len(mins)+1)*0.5)
        else :
            outside_ratios.append(0)
    print(outside_ratios)
    return torch.mean(torch.Tensor(outside_ratios,dtype=torch.float32))

"""
loss function for binary classification
"""
class Consistency_loss_wrapper(nn.Module):
    """Some Information about Consistency_loss_wrapper"""
    def __init__(self,args):
        super(Consistency_loss_wrapper, self).__init__()
        if args.loss == 'crossentropy':
            self.lossfun = Binary_Loss_wrapper(args)
        elif args.loss == "focal":
            self.lossfun = FocalLoss(logits=True)
        elif args.loss =="dicefocal":
            self.lossfun = DiceFocalLoss()
        else:
            print('error')
            exit()
        self.consistency_loss = nn.MSELoss()
        
    def forward(self, pred, targets):
        x,aug_x,coordinate = pred
        i,j,h,w = coordinate
        loss = 0

        loss += self.lossfun(x,targets)
        aug_targets = TF.resized_crop(targets, i, j, h, w, 256)
        loss += self.lossfun(aug_x,aug_targets)
        x = TF.resized_crop(x, i, j, h, w, 256)
        loss += self.consistency_loss(x,aug_x)
        
        return loss

class Loss_wrapper(nn.Module):
    
    def __init__(self,args):
        super(Loss_wrapper, self).__init__()
        self.lossfun = None
        self.args = args
        weight = torch.ones((2),device='cuda')
        weight[1]*=args.classweight
        if args.loss == 'crossentropy':
            self.lossfun = nn.CrossEntropyLoss(weight=weight)
        elif args.loss == 'dicecrossentropy':
            self.lossfun = nn.CrossEntropyLoss(weight=weight)
            self.dicelossfun = DiceLoss()

    def forward(self, pred, target):
        if torch.__version__ != '1.10.1':
            target = torch.argmax(target,dim=1)
        loss = self.lossfun(pred,target)
        if self.args.loss == 'dicecrossentropy':
            loss+=self.dicelossfun(pred,target)
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

    def forward(self, inputs, targets):
        loss = 0
        for i in range(self.num_classes):
            loss+=self.loss(inputs[:,i],targets[:,i])
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

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        
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

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = self.focal_loss(inputs, targets)
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE