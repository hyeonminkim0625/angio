import argparse
import json
from random import shuffle
import torch
import torch.nn as nn
from pathlib import Path

from models.model import BaseLine_wrapper
from models.loss import Loss_wrapper, Binary_Loss_wrapper
from engine import train_one_epoch, evaluate
from models.unet_plusplus import Nested_UNet

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
from datasets.dataset import Angio_Dataset
import wandb
import os
import numpy as np
import random
import torch_optimizer as optim
from optimizer import radam_lookahead as rl

def get_args_parser():
    parser = argparse.ArgumentParser('Set Segmentation model', add_help=False)
    #train
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=32, type=int )
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--multigpu', action='store_true')
    parser.add_argument('--rank', default="1", type=str)
    parser.add_argument('--loss', default="crossentropy", type=str)
    parser.add_argument('--opt', default="rll", type=str)
    parser.add_argument('--img_size', default=512, type=int)
    parser.add_argument('--withcoordinate', default='concat', type=str)
    parser.add_argument('--classweight', default=1.0, type=float)
    parser.add_argument('--histogram_eq', default=False,type=bool)
    parser.add_argument('--sigma', default=0.3, type=float)
    parser.add_argument('--valperepoch', default=2, type=int)
    parser.add_argument('--model',default="unet",type=str)
    parser.add_argument('--num_classes',default=2, type=int)
    parser.add_argument('--weight_dir', default='./weight', help='path where to save, empty for no saving')
    
    
    #eval
    parser.add_argument('--output_dir', default='./result', help='sample prediction, ground truth')
    parser.add_argument('--mode',default='train',type=str)
    parser.add_argument('--mask_argmax', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--weight_path',default="/",type=str)
    parser.add_argument('--saveallfig', action='store_true')
    parser.add_argument('--onlymask', action='store_true')
    parser.add_argument('--report_hard_sample', default=0, type=int)

    return parser

def train(args):
    print(args)
    
    model = None

    if args.model == 'unet' or args.model == 'deeplab' or args.model == 'fcn' or args.model == 'unet' or args.model == 'unetpp' or args.model == "deeplabv3plus":
        model = BaseLine_wrapper(args)
    else:
        print("model input error")
        exit()

    criterion = None
    if args.loss == 'crossentropy' or args.loss == 'dicecrossentropy':
        criterion = Loss_wrapper(args)
    elif args.loss == 'focal' or args.loss == 'dicefocal':
        criterion = Binary_Loss_wrapper(args)
    else:
        print("loss input error")
        exit()
    
    if args.multigpu:
        model = nn.DataParallel(model)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]= args.rank

    device = torch.device("cuda")

    optimizer = None
    base_opt = None
    if args.opt == 'rll':
        base_opt=rl.Ralamb(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        optimizer = rl.Lookahead(base_opt,alpha=0.5,k=5)
    elif args.opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'radam':
        optimizer = optim.RAdam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)
    model.to(device)
    criterion.to(device)

    
    train_dataset = Angio_Dataset(args.num_classes,mode = "train",args=args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,num_workers=16, batch_size=args.batch_size,shuffle=True,drop_last=False)

    val_dataset = Angio_Dataset(args.num_classes,mode = "val",args=args)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,num_workers=16, batch_size=args.batch_size,shuffle=False,drop_last=False)

    for i in range(args.epochs):
        
        train_one_epoch(model, criterion, train_dataloader , optimizer ,device ,args=args)

        if (i+1)%args.valperepoch==0:
            evaluate(model, criterion, val_dataloader ,device , args)
            weight_dict = {
                'epoch': i,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()}
            
            if base_opt is not None:
                weight_dict['base_optimizer_state_dict'] = base_opt.state_dict()
            torch.save(weight_dict,
                args.weight_dir+'/'+args.model+'_'+str(i)+'.pth')
            
        scheduler.step()

def eval(args):
    model = None
    if args.model == 'unet' or args.model == 'deeplab' or args.model == 'unet' or args.model == 'unetpp' or args.model == "deeplabv3plus":
        model = BaseLine_wrapper(args)
    else:
        print("model input error")
        exit()

    criterion = None
    if args.loss == 'crossentropy':
        criterion = Loss_wrapper(args)
    elif args.loss == 'focal' or args.loss == 'dicefocal':
        criterion = Binary_Loss_wrapper(args)
    else:
        print("loss input error")
        exit()

    if args.multigpu:
        model = nn.DataParallel(model)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]= args.rank
    
    device = torch.device("cuda")

    model.to(device)
    criterion.to(device)

    val_dataset = Angio_Dataset(args.num_classes,mode = args.mode,args=args)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,num_workers=16, batch_size=args.batch_size,drop_last=False)

    if args.weight_path is not "/":
        checkpoint = torch.load(args.weight_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    evaluate(model, criterion, val_dataloader ,device , args)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Segmentation training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    random_seed = 777
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if args.multigpu:
        torch.cuda.manual_seed_all(random_seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
   

    if args.wandb:
        wandb.init(project='angio')#entity="medi-whale"
    if args.output_dir:
        if args.eval:
            Path(args.output_dir+'_'+args.model+'_'+args.mode).mkdir(parents=True, exist_ok=True)
            Path(args.output_dir+'_'+args.model+'_'+args.mode+'/hard_sample').mkdir(parents=True, exist_ok=True)
        else:
            for i in range(100):
                #if args.wandb:
                    #args.model = wandb.config['model']
                if not Path(args.weight_dir+'_'+args.model+'_'+str(i)).is_dir():
                    args.weight_dir = args.weight_dir+'_'+args.model+'_'+str(i)
                    wandb.config['weight_dir']  = args.weight_dir
                    Path(args.weight_dir).mkdir(parents=True, exist_ok=True)
                    break
    if args.eval:
        eval(args)
    else:
        """
        wandb.config['multigpu']=True
        wandb.config['num_classes']=2
        wandb.config['output_dir']='.'
        wandb.config['mode']='train'
        wandb.config['mask_argmax']=True
        wandb.config['eval']=False
        wandb.config['weight_path']='/'
        wandb.config['saveallfig']=False
        wandb.config['onlymask']=False
        wandb.config['report_hard_sample']=0
        wandb.config['wandb']=True
        """

        train(args)
"""
보류
"""
def run(gpu, ngpus_per_node ,args):

    dist.init_process_group(backend="gloo",rank=gpu,world_size=ngpus_per_node)

    model = BaseLine_model(False, args.num_classes)
    criterion = Loss_wrapper(args)

    
    model = model.to(gpu)
    ddp_model = DistributedDataParallel(model,device_ids=[gpu])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr = args.lr)
    criterion.to(gpu)

    dataset = IVUS_Dataset(args.num_classes)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,batch_sampler = sampler,  batch_size=args.batch_size)

    for i in range(100):
        train_one_epoch(model, criterion, dataloader , optimizer ,gpu , i)
        evaluate(model, criterion, dataloader ,gpu , i)