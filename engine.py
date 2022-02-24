import math
import os
import sys
from typing import Iterable
from torch._C import dtype
from tqdm import tqdm
import torchvision
import wandb
import numpy as np
import torch
from shutil import copyfile
import cv2
import pdb
import pandas as pd
import pickle
from metric import averaged_hausdorff_distance as ahd, calculate_iou, calculate_overlab_contour, compute_mean_iou
from scipy.spatial.distance import directed_hausdorff
import torch.nn.functional as F

from models.loss import centerline_loss_fn, vector_loss


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, args, scheduler):
    
    model.train()
    criterion.train()
    total_loss = 0
    batch_num = len(data_loader)
    for samples, targets, _ in tqdm(data_loader):

        samples = samples.to(device)

        targets_index = torch.stack([s.to(device) for s in targets["index"]],dim=0)
        if 'center' in targets.keys():
            targets_center = torch.stack([s.to(device) for s in targets["center"]],dim=0)
        if 'coord' in targets.keys():
            targets_coord = torch.stack([s.to(device) for s in targets["coord"]],dim=0)

        outputs = model(samples)
        loss = criterion(outputs, targets_index)
        
        total_loss += float(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
    
    total_loss = total_loss/(batch_num*args.batch_size)
    if args.wandb:
        wandb.log({"train average losses" : total_loss})

@torch.no_grad()
def evaluate(model, criterion, data_loader, device, args):
    
    model.eval()
    criterion.eval()

    total_loss = 0
    batch_num = len(data_loader)

    path_iou_array=[]
    
    for samples, targets,paths in tqdm(data_loader):
        
        samples = samples.to(device)

        targets_index = torch.stack([s.to(device) for s in targets["index"]],dim=0)
        if 'centerline_distancemap' in targets.keys():
            targets_center = torch.stack([s.to(device) for s in targets["centerline_distancemap"]],dim=0)
        
        outputs = model(samples)
        loss = criterion(outputs, targets_index)
         
        total_loss += float(loss)

        num_classes = outputs.shape[1]

        #calculate by batch
        for j in range(samples.shape[0]):

            """
            convert probability tensor to mask tensor
            """
            output_mask = None
            target_mask = None
            info_dictionary = {"path":paths[j].split('.')[0]}

            if args.mask_argmax:
                # [h,w,class] -> [class,h,w]
                output_mask = F.one_hot(torch.argmax(outputs[j],dim=0),num_classes=args.num_classes).permute(2,0,1)
                target_mask = F.one_hot(torch.argmax(targets_index[j],dim=0),num_classes=args.num_classes).permute(2,0,1)

            else:
                pass

            iou = calculate_iou(output_mask,target_mask,args.num_classes)

            info_dictionary["class1_iou"]=float(iou[1])
            info_dictionary["class0_iou"]=float(iou[0])
            
            path_iou_array.append(
                info_dictionary
            )

            """
            save pic, optional
            """
            if args.saveallfig:
                if args.mask_argmax:
                # [h,w,class] -> [class,h,w]
                    output_mask = F.one_hot(torch.argmax(outputs[j],dim=0),num_classes=args.num_classes).permute(2,0,1)
                    target_mask = F.one_hot(torch.argmax(targets_index[j],dim=0),num_classes=args.num_classes).permute(2,0,1)
                else:
                    pass
                output_mask = output_mask.to(dtype=torch.bool, device='cpu')
                target_mask = target_mask.to(dtype=torch.bool, device='cpu')
                img = None

                if args.onlymask:
                    img = torch.zeros_like(samples[j],dtype=torch.uint8,device='cpu')
                else:
                    img = (outputs[j]*255).to(dtype=torch.uint8,device='cpu');

                pred_file_name = args.output_dir+'_'+args.model+'_'+args.mode+'/'+paths[j].split('.')[0]+'_pred.png'
                target_file_name =  args.output_dir+'_'+args.model+'_'+args.mode+'/'+paths[j].split('.')[0]+'_target.png'
                data_path = '/data/angiosegmentation/raw_img/a-'+paths[j]+'.jpg'

                """
                prediction mask
                """
                if torch.__version__ != '1.8.1+cu101' :
                    """
                    not support '1.8.1+cu101'
                    """
                    saveimg = torchvision.utils.draw_segmentation_masks(img,output_mask[1:],colors=[(255,0,51),(102,255,102)])
                    torchvision.utils.save_image(saveimg/255.0 ,pred_file_name)

                """
                target mask
                """
                if torch.__version__ != '1.8.1+cu101' :
                    """
                    not support '1.8.1+cu101'
                    """
                    saveimg = torchvision.utils.draw_segmentation_masks(img,target_mask[1:],colors=[(255,0,51),(102,255,102)])
                    torchvision.utils.save_image(saveimg/255.0 , target_file_name)

                """
                save
                1. input data
                2. target mask img
                3. prediction mask img
                4. target mask array npy
                5. prediction mask array npy
                """

                copyfile(data_path,"./"+args.output_dir+'_'+args.model+'_'+args.mode+'/'+paths[j]+'.jpg')

                np.save(pred_file_name.replace('png','npy'),output_mask)
                np.save(target_file_name.replace('png','npy'),target_mask)
    
    if args.report_hard_sample != 0:
        with open(args.output_dir+'_'+args.model+'_'+args.mode+'/'+'iou_array.npy','wb') as f:
            pickle.dump(path_iou_array,f)
        for i in range(1,num_classes):
            path_iou_array = sorted(path_iou_array, key= lambda p : (p['class'+str(i)+'_iou']))
            topk_path_iou_array = path_iou_array[:args.report_hard_sample]
            for j in topk_path_iou_array:

                pred_file_name = args.output_dir+'_'+args.model+'_'+args.mode+'/'+j['path']+'_pred.png'
                target_file_name =  args.output_dir+'_'+args.model+'_'+args.mode+'/'+j['path']+'_target.png'
                
                if torch.__version__ != '1.8.1+cu101' :
                    """
                    not support '1.8.1+cu101'
                    """
                    copyfile(pred_file_name,args.output_dir+'_'+args.model+'_'+args.mode+'/hard_sample/'+j['path']+'_pred.png')
                    copyfile(target_file_name,args.output_dir+'_'+args.model+'_'+args.mode+'/hard_sample/'+j['path']+'_target.png')

                copyfile(pred_file_name.replace('png','npy'),args.output_dir+'_'+args.model+'_'+args.mode+'/hard_sample/'+j['path']+'_pred.npy')
                copyfile(target_file_name.replace('png','npy'),args.output_dir+'_'+args.model+'_'+args.mode+'/hard_sample/'+j['path']+'_target.npy')
                copyfile(args.output_dir+'_'+args.model+'_'+args.mode+"/"+j['path']+".jpg",args.output_dir+'_'+args.model+'_'+args.mode+"/hard_sample/"+j['path']+".jpg")
                
                if args.wandb:
                    bg_img = cv2.imread(args.output_dir+'_'+args.model+'_'+args.mode+"/hard_sample/"+j['path']+".jpg")

                    bg_img = cv2.resize(bg_img,(args.img_size,args.img_size))

                    pred_mask = np.load(pred_file_name.replace('png','npy'))
                    target_mask = np.load(target_file_name.replace('png','npy'))
                    pred_mask = np.argmax(pred_mask,axis=0)
                    target_mask = np.argmax(target_mask,axis=0)

                    labels = {
                    1: "class 1",
                    }

                    temp_dict = {}
                    temp_dict["hard_sample"] = wandb.Image(bg_img,caption=f"class 1 iou : {j['class1_iou']*100},  {j['path']}", masks={
                    "prediction" : {"mask_data" : pred_mask, "class_labels" : labels},
                    "ground truth" : {"mask_data" : target_mask, "class_labels" : labels}})

                    wandb.log(temp_dict)

    total_loss = total_loss/(batch_num*args.batch_size)
   
    if args.wandb:
        wandb_dict = {"eval "+str(args.mode)+" average losses" : total_loss}

        for i in range(0,num_classes):
            temp = np.array([p['class'+str(i)+'_iou'] for p in path_iou_array])
            wandb_dict["class"+str(i)+" iou"] = np.mean(temp)*100
        
        wandb_dict["total iou"] = (wandb_dict['class0 iou'] + wandb_dict['class1 iou'])/2.0

        wandb.log(wandb_dict)
    if args.eval:
        for i in range(1,num_classes):
            temp = np.array([p['class'+str(i)+'_iou'] for p in path_iou_array])
            print("class"+str(i)+" iou", np.mean(temp)*100)