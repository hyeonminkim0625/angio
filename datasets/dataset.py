import torch
import torchvision
from torch.utils.data import Dataset
from glob import glob
import PIL.Image
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import albumentations as A
import cv2
import os
import csv
import pdb
import pandas as pd
from tqdm import tqdm
from utils import histogram_eq

def gaussian_heatmap(sigma: int, spread: int):
    extent = int(spread * sigma)
    center = spread * sigma / 2
    heatmap = np.zeros([extent, extent], dtype=np.float32)
    for i_ in range(extent):
        for j_ in range(extent):
            heatmap[i_, j_] = 1 / 2 / np.pi / (sigma ** 2) * np.exp(
                -1 / 2 * ((i_ - center - 0.5) ** 2 + (j_ - center - 0.5) ** 2) / (sigma ** 2))
    heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
    return heatmap

def gaussian_heatmap_re(heatmap,x,y):
    for i_ in range(512):
        for j_ in range(512):
            heatmap[i_, j_] += ((y-i_)**2 + (x-j_)**2)**0.2
    return heatmap

class Angio_Dataset(torch.utils.data.Dataset):
    def __init__(self,num_classes,mode,args):
        super(Angio_Dataset, self).__init__()
        self.image_path = None
        self.args = args

        self.angio_list = pd.read_csv('./angio_list.csv')
        """
        roi == 1
        """

        not_use = [1336,275,228,248,712,631,268,4,180,1364]
        not_use = [str(p) for p in not_use]
         
        img_list = []
        if mode == "train":
            for i in self.angio_list.iterrows():
                if i[1]['train']==1:
                    if i[1]['origin'].split('-')[1].split('.')[0] not in not_use:
                        temp = ['/data/angiosegmentation/raw_img/'+i[1]['origin'], '/data/angiosegmentation/mask_correct/'+i[1]['segmentation'],(i[1]['x1'], i[1]['y1'], i[1]['x2'], i[1]['y2'])]
                        img_list.append(temp)
                  
            self.image_path = img_list
            self.mode = "train"
            print('train ',len(self.image_path))

        elif mode == "val":
            for i in self.angio_list.iterrows():
                if i[1]['train']==0:
                    if i[1]['origin'].split('-')[1].split('.')[0] not in not_use:
                        temp = ['/data/angiosegmentation/raw_img/'+i[1]['origin'], '/data/angiosegmentation/mask_correct/'+i[1]['segmentation'],(i[1]['x1'], i[1]['y1'], i[1]['x2'], i[1]['y2'])]
                        img_list.append(temp)                
                  
            self.image_path = img_list
            self.mode = "val"
            print('val ',len(self.image_path))
        else:
            print('error')
            exit()

        self.num_classes = num_classes
        if self.mode == "train":
            self.transform = make_transform(args)
        else:
            self.transform = make_transform_val(args)
        self.resnet_mean = [0.485, 0, 0.0]
        self.resnet_std = [0.229, 1.0, 1.0]
        
    def __getitem__(self, index):

        def img_load(path):
            im = cv2.imread(path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            return im
       
        image_path = self.image_path[index][0]
        target_index = img_load(self.image_path[index][1])

        target_index = target_index[:,:,0]
        h,w = target_index.shape
        target = np.zeros((h,w,self.num_classes))
        for i in range(self.num_classes):
            target[:,:,i] = (target_index==i)*1
        target = target.astype(np.float32)
        img = img_load(image_path)

        #img[:,:,1] = histogram_eq(img[:,:,1])
        if self.args.withcoordinate=='concat':
            x1, y1, x2, y2 = self.image_path[index][2]
            annotated_dot = np.zeros((512,512))

            #annotated_dot = gaussian_heatmap_re(annotated_dot,x1,y1)
            #annotated_dot = gaussian_heatmap_re(annotated_dot,x2,y2)
            #annotated_dot = (annotated_dot / np.max(annotated_dot) * 255).astype(np.uint8)
            #annotated_dot = 255-annotated_dot
            annotated_dot = np.zeros((512,512))
            annotated_dot[int(x1),int(y1)]=255# y1 x1
            annotated_dot[int(x2),int(y2)]=255

            annotated_dot = cv2.GaussianBlur(annotated_dot,(15,15),0)*30

            img[:,:,2] = annotated_dot

        elif self.args.withcoordinate=='add':
            x1, y1, x2, y2 = self.image_path[index][2]
            img = cv2.circle(img,(int(x1),int(y1)),5,(255,255,255),thickness=-1)
            img = cv2.circle(img,(int(x2),int(y2)),5,(255,255,255),thickness=-1)
        else:
            """
            no use
            """
            pass
        if self.mode == "train":
            transformed = self.transform(image=img, mask=target)
            target = TF.to_tensor(transformed['mask'])
            img = TF.to_tensor(transformed['image'])
            img = TF.normalize(img,mean=self.resnet_mean, std=self.resnet_std)

        else:
            transformed = self.transform(image=img, mask=target)

            target = TF.to_tensor(transformed['mask'])
            img = TF.to_tensor(transformed['image'])
            img = TF.normalize(img,mean=self.resnet_mean, std=self.resnet_std)

        #return img target patient num
        return img , target, image_path.split('/')[4].split('-')[1].split('.')[0].split('_')[0]

    def __len__(self):
        return len(self.image_path)

def make_transform(args):
    transform = A.Compose([
    A.Resize(width=args.img_size, height=args.img_size),
    A.RandomResizedCrop(width=args.img_size, height=args.img_size, scale=(0.8,1.0),p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    #A.RandomBrightnessContrast(brightness_limit=0.1,contrast_limit=0.1,p=0.5),#option1
    #A.Equalize(),
    ])
    return transform

def make_transform_val(args):
    transform = A.Compose([
    A.Resize(width=args.img_size, height=args.img_size),
    ])
    return transform