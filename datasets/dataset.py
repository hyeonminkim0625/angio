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
from utils import histogram_eq, gaussian_heatmap_re, draw_centerline
from pathlib import Path


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
        if not os.path.exists("/data/angiosegmentation/heatmap/"):
            Path('/data/angiosegmentation/heatmap/').mkdir(parents=True, exist_ok=True)

        if mode == "train":
            for i in tqdm(self.angio_list.iterrows()):
                if i[1]['train']==1:
                    if i[1]['origin'].split('-')[1].split('.')[0] not in not_use:
                        temp = ['/data/angiosegmentation/raw_img/'+i[1]['origin'],
                        '/data/angiosegmentation/mask_correct/'+i[1]['segmentation'],
                        (i[1]['x1'], i[1]['y1'], i[1]['x2'], i[1]['y2']),
                        i[1]['origin'].split('-')[1].split('.')[0]]
                        if not os.path.exists("/data/angiosegmentation/heatmap/"+str(i[1]['origin'].split('-')[1].split('.')[0])+'_'+str(self.args.sigma)+'.npy'):
                            x1, y1, x2, y2 = temp[2]
                            annotated_dot = np.zeros((512,512))

                            annotated_dot = gaussian_heatmap_re(annotated_dot,x1,y1,self.args.sigma)
                            annotated_dot = gaussian_heatmap_re(annotated_dot,x2,y2,self.args.sigma)
                            annotated_dot = (annotated_dot / np.max(annotated_dot) * 255).astype(np.uint8)
                            annotated_dot = 255-annotated_dot
                            np.save("/data/angiosegmentation/heatmap/"+str(i[1]['origin'].split('-')[1].split('.')[0])+'_'+str(self.args.sigma)+'.npy',annotated_dot)

                        if not os.path.exists("/data/angiosegmentation/centerline/"+str(i[1]['origin'].split('-')[1].split('.')[0])+'.png'):
                            centerline_img = draw_centerline('/data/angiosegmentation/mask_correct/'+i[1]['segmentation'])
                            cv2.imwrite("/data/angiosegmentation/centerline/"+str(i[1]['origin'].split('-')[1].split('.')[0])+'.png',centerline_img)
                        img_list.append(temp)

            self.image_path = img_list
            self.mode = "train"
            print('train ',len(self.image_path))

        elif mode == "val":
            for i in tqdm(self.angio_list.iterrows()):
                if i[1]['train']==0:
                    if i[1]['origin'].split('-')[1].split('.')[0] not in not_use:
                        temp = ['/data/angiosegmentation/raw_img/'+i[1]['origin'],
                        '/data/angiosegmentation/mask_correct/'+i[1]['segmentation'],
                        (i[1]['x1'], i[1]['y1'], i[1]['x2'], i[1]['y2']),
                        i[1]['origin'].split('-')[1].split('.')[0]]
                        if not os.path.exists("/data/angiosegmentation/heatmap/"+str(i[1]['origin'].split('-')[1].split('.')[0])+'_'+str(self.args.sigma)+'.npy'):
                            x1, y1, x2, y2 = temp[2]
                            annotated_dot = np.zeros((512,512))

                            annotated_dot = gaussian_heatmap_re(annotated_dot,x1,y1,self.args.sigma)
                            annotated_dot = gaussian_heatmap_re(annotated_dot,x2,y2,self.args.sigma)
                            annotated_dot = (annotated_dot / np.max(annotated_dot) * 255).astype(np.uint8)
                            annotated_dot = 255-annotated_dot
                            np.save("/data/angiosegmentation/heatmap/"+str(i[1]['origin'].split('-')[1].split('.')[0])+'_'+str(self.args.sigma)+'.npy',annotated_dot)

                        if not os.path.exists("/data/angiosegmentation/centerline/"+str(i[1]['origin'].split('-')[1].split('.')[0])+'.png'):
                            centerline_img = draw_centerline('/data/angiosegmentation/mask_correct/'+i[1]['segmentation'])
                            cv2.imwrite("/data/angiosegmentation/centerline/"+str(i[1]['origin'].split('-')[1].split('.')[0])+'.png',centerline_img)
                        
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
        
        if self.args.histogram_eq:
            self.resnet_mean = [0.485, 0.0, 0.0]
            self.resnet_std = [0.229, 1.0, 1.0]
        else:
            self.resnet_mean = [0.485, 0.456, 0.0]
            self.resnet_std = [0.229, 0.224, 1.0]
        
    def __getitem__(self, index):

        def img_load(path):
            im = cv2.imread(path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            return im
       
        image_path = self.image_path[index][0]
        target_index = img_load(self.image_path[index][1])

        target_index = target_index[:,:,0]
        h,w = target_index.shape
        
        
        img = img_load(image_path)

        if self.args.centerline:
            centerline = cv2.imread("/data/angiosegmentation/centerline/"+self.image_path[index][3]+'.png')
            centerline = cv2.cvtColor(centerline, cv2.COLOR_BGR2GRAY)
            target = np.zeros((h,w,self.num_classes+1))
            target[:,:,-1] = centerline
        else:
            target = np.zeros((h,w,self.num_classes))

        for i in range(self.num_classes):
            target[:,:,i] = (target_index==i)*1
        target = target.astype(np.float32)

        if self.args.histogram_eq:
            img[:,:,1] = histogram_eq(img[:,:,1])

        if self.args.withcoordinate=='concat_filter':
            x1, y1, x2, y2 = self.image_path[index][2]
            annotated_dot = np.zeros((512,512))
            annotated_dot[int(y1),int(x1)]=255# y1 x1
            annotated_dot[int(y2),int(x2)]=255
            annotated_dot = cv2.GaussianBlur(annotated_dot,(15,15),0)*10
            img[:,:,2] = annotated_dot
        
        if self.args.withcoordinate=='concat_point':
            x1, y1, x2, y2 = self.image_path[index][2]
            annotated_dot = np.zeros((512,512,3))
            annotated_dot = cv2.circle(annotated_dot,(int(x1),int(y1)),5,(255,255,255),thickness=-1)
            annotated_dot = cv2.circle(annotated_dot,(int(x2),int(y2)),5,(255,255,255),thickness=-1)

            img[:,:,2] = annotated_dot[:,:,0]
        
        if self.args.withcoordinate=='concat_heatmap':
            annotated_dot = np.load("/data/angiosegmentation/heatmap/"+self.image_path[index][3]+'_'+str(self.args.sigma)+'.npy')
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

        target_dict = {"index": target}
        if self.args.centerline:
            target_dict['index'] = target[:-1]
            target_dict['center'] = target[-1]
        #if self.args.vectorloss:
        #    target_dict['coord'] = torch.tensor(self.image_path[index][2])
        
        return img , target_dict, image_path.split('/')[4].split('-')[1].split('.')[0].split('_')[0]
       
        #reage_path.split('/')[4].split('-')[1].split('.')[0].split('_')[0]turn img target patient num
            

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