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
        self.transform = make_transform()
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
            target = TF.to_tensor(target)
            img = TF.to_tensor(img)
            img = TF.normalize(img,mean=self.resnet_mean, std=self.resnet_std)


        #return img target patient num
        return img , target, image_path.split('/')[4].split('-')[1].split('.')[0].split('_')[0]

    def __len__(self):
        return len(self.image_path)

class IVUS_Dataset(torch.utils.data.Dataset):
    def __init__(self,num_classes,mode,args):
        super(IVUS_Dataset, self).__init__()
        self.image_path = None
        self.IVUS_frame = pd.read_csv("IVUS_frames.csv").to_numpy()
        self.IVUS_frame = self.IVUS_frame[self.IVUS_frame[:,4]==1]
        self.args = args
        """
        roi == 1
        """

        if not os.path.exists("./trainset.npy"):
            
            self.image_path = glob("/data/IVUS_extract/**/roi_image/*.png")
            
            patient_list = ["F000"+str(i) for i in range(1,10)]+["F00"+str(i) for i in range(10,100)]+["F0"+str(i) for i in range(100,1000)]+["F"+str(i) for i in range(1000,1301)]
            random.shuffle(patient_list)

            trainset_list = patient_list[: int(len(patient_list)*0.8)]
            validation_list = patient_list[int(len(patient_list)*0.8): int(len(patient_list)*0.9)]
            testset_list = patient_list[int(len(patient_list)*0.9):]

            np.save('./train_patient_list.npy',np.array(trainset_list))
            np.save('./validation_patient_list.npy',np.array(validation_list))
            np.save('./test_patient_list.npy',np.array(testset_list))
            
            trainset_list = np.load('./train_patient_list.npy')
            validation_list = np.load('./validation_patient_list.npy')
            testset_list = np.load('./test_patient_list.npy')
            temp=[]

            remove_patient_list = ["F0"+str(i) for i in range(201,261)]

            print("make train dataset...")
            for i in tqdm(trainset_list):
                if i in remove_patient_list:
                    continue
                for j in self.IVUS_frame[self.IVUS_frame[:,1]==i][:,2]:
                    temp.append("/data/IVUS_extract/"+i+"/roi_image/"+str(j)+".png")
            trainset_list = temp

            print("make validation dataset...")
            temp=[]
            for i in tqdm(validation_list):
                if i in remove_patient_list:
                    continue
                for j in self.IVUS_frame[self.IVUS_frame[:,1]==i][:,2]:
                    temp.append("/data/IVUS_extract/"+i+"/roi_image/"+str(j)+".png")
            validation_list = temp

            print("make test dataset...")
            temp=[]
            for i in tqdm(testset_list):
                if i in remove_patient_list:
                    continue
                for j in self.IVUS_frame[self.IVUS_frame[:,1]==i][:,2]:
                    temp.append("/data/IVUS_extract/"+i+"/roi_image/"+str(j)+".png")
            testset_list = temp

            np.save('./trainset.npy',np.array(trainset_list))
            np.save('./validationset.npy',np.array(validation_list))
            np.save('./testset.npy',np.array(testset_list))
        

        if mode == "train":
            path_list = np.load("./trainset.npy")
            image_path = []
            for p in path_list:
                image_dict = {"anchor" : p ,"support" : []}
                frame_num = p.split('/')[-1].split('.')[0]

                for k in range(-args.frame,0):
                    """
                    -n,,, -1 frame
                    """
                    if os.path.isfile(p.replace('roi_image','origin').replace(frame_num,str(int(frame_num)+k))):
                        image_dict['support'].append(p.replace('roi_image','origin').replace(frame_num,str(int(frame_num)+k)))
                
                if len(image_dict['support'])==args.frame:
                    image_path.append(image_dict)
            self.image_path = image_path

            self.mode = "train"

        elif mode == "val":
            path_list = np.load("./validationset.npy")
            image_path = []
            for p in path_list:
                image_dict = {"anchor" : p ,"support" : []}
                frame_num = p.split('/')[-1].split('.')[0]

                for k in range(-args.frame,0):
                    """
                    -n,,, -1 frame
                    """
                    if os.path.isfile(p.replace('roi_image','origin').replace(frame_num,str(int(frame_num)+k))):
                        image_dict['support'].append(p.replace('roi_image','origin').replace(frame_num,str(int(frame_num)+k)))

                if len(image_dict['support'])==args.frame:
                    image_path.append(image_dict)
            self.image_path = image_path
            self.mode = "val"
        elif mode == "test":
            path_list = np.load("./testset.npy")
            image_path = []
            for p in path_list:
                image_dict = {"anchor" : p ,"support" : []}
                frame_num = p.split('/')[-1].split('.')[0]

                for k in range(-args.frame,0):
                    """
                    -n,,, -1 frame
                    """
                    if not os.path.isfile(p.replace('roi_image','origin').replace(frame_num,str(int(frame_num)+k))):
                        image_dict['support'].append(p.replace('roi_image','origin').replace(frame_num,str(int(frame_num)+k)))

                if len(image_dict['support'])==args.frame:
                    image_path.append(image_dict)
            self.image_path = image_path
            self.mode = "test"

        self.num_classes = num_classes
        self.transform = make_transform()
        self.resnet_mean = [0.485, 0.456, 0.406]
        self.resnet_std = [0.229, 0.224, 0.225]
        
    def __getitem__(self, index):

        def img_load(path):
            im = cv2.imread(path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            return im
       
        anchor_image_path = self.image_path[index]["anchor"]

        target_index = np.load(anchor_image_path.replace('roi_image','roi_mask').replace('png','npy'))
        h,w = target_index.shape
        target = np.zeros((h,w,self.num_classes))
        for i in range(self.num_classes):
            target[:,:,i] = (target_index==i)*1
        target = target.astype(np.float32)
        anchor_img = img_load(anchor_image_path)

        support_set = []

        if self.mode == "train":
            transformed = self.transform(image=anchor_img, mask=target)
            target = TF.to_tensor(transformed['mask'])
            anchor_img = TF.to_tensor(transformed['image'])
            anchor_img = TF.normalize(anchor_img,mean=self.resnet_mean, std=self.resnet_std)

            for p in self.image_path[index]["support"]:
                support_img = img_load(p)

                transformed = self.transform(image=support_img)

                support_img = TF.to_tensor(transformed['image'])
                support_img = TF.normalize(support_img,mean=self.resnet_mean, std=self.resnet_std)
                support_set.append(support_img)

        else:
            target = TF.to_tensor(target)
            anchor_img = TF.to_tensor(anchor_img)
            anchor_img = TF.normalize(anchor_img,mean=self.resnet_mean, std=self.resnet_std)

            for p in self.image_path[index]["support"]:
                support_img = img_load(p)

                support_img = TF.to_tensor(support_img)
                support_img = TF.normalize(support_img,mean=self.resnet_mean, std=self.resnet_std)
                support_set.append(support_img)

        return {"anchor": anchor_img, "support" : support_set} , target, anchor_image_path.split('/')[3]+'_'+anchor_image_path.split('/')[5]

    def __len__(self):
        return len(self.image_path)

def make_transform():
    transform = A.Compose([
    A.RandomResizedCrop(width=256, height=256, scale=(0.8,1.0),p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    ])
    return transform