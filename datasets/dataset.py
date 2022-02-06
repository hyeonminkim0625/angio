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

class Angio_Dataset(torch.utils.data.Dataset):
    def __init__(self,num_classes,mode,args):
        super(Angio_Dataset, self).__init__()
        self.image_path = None
        self.args = args
        """
        roi == 1
        """

        if not os.path.exists("./trainset.npy"):
            
            self.image_path = glob("/data/angiosegmentation/raw_img/*.jpg")
            random.shuffle(self.image_path)
            
            trainset_list = self.image_path[: int(len(self.image_path)*0.8)]
            validation_list = self.image_path[int(len(self.image_path)*0.8): int(len(self.image_path)*0.9)]
            testset_list = self.image_path[int(len(self.image_path)*0.9):]

            np.save('./trainset.npy',np.array(trainset_list))
            np.save('./validationset.npy',np.array(validation_list))
            np.save('./testset.npy',np.array(testset_list))

        if mode == "train":
            path_list = np.load("./trainset.npy")
            self.image_path = [[i,'/data/angiosegmentation/mask_correct/b-'+i.split('-')[1].replace('.jpg','_M.png')] for i in path_list]
            self.mode = "train"

        elif mode == "val":
            path_list = np.load("./validationset.npy")
            self.image_path = [[i,'/data/angiosegmentation/mask_correct/b-'+i.split('-')[1].replace('.jpg','_M.png')] for i in path_list]
            self.mode = "val"
        elif mode == "test":
            path_list = np.load("./testset.npy")
            self.image_path = [[i,'/data/angiosegmentation/mask_correct/b-'+i.split('-')[1].replace('.jpg','_M.png')] for i in path_list]
            self.mode = "test"

        self.num_classes = num_classes
        if self.mode == "train":
            self.transform = make_transform(args)
        else:
            self.transform = make_transform_val(args)
        self.resnet_mean = [0.485, 0.456, 0.406]
        self.resnet_std = [0.229, 0.224, 0.225]
        
    def __getitem__(self, index):

        def img_load(path):
            im = cv2.imread(path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            return im
       
        image_path = self.image_path[index][0]

        target_index = img_load(self.image_path[index][1])

        h,w = target_index.shape
        target = np.zeros((h,w,self.num_classes))
        for i in range(self.num_classes):
            target[:,:,i] = (target_index==i)*1
        target = target.astype(np.float32)
        img = img_load(image_path)

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
    ])
    return transform

def make_transform_val(args):
    transform = A.Compose([
    A.Resize(width=args.img_size, height=args.img_size),
    ])
    return transform