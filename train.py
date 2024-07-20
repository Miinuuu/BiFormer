import os
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from Trainer import Model
import hashlib
from pathlib import Path
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio
from torch.utils.data.distributed import DistributedSampler
from model.scheduler import get_learning_rate
import wandb as wandb
from functools import partial
import torch.nn as nn
from typing import List, Optional, Sequence, Union
from tqdm import tqdm
from  datamodules.vimeo_triplet import vimeo_triplet
from  datamodules.vimeo_setuplet import vimeo_setuplet
from utils import load_yml2args, text_color

device = torch.device("cuda")
exp = os.path.abspath('.').split('/')[-1]


def train(model, args ,epoch=0,global_step=0,cur_psnr=0):

    start_epoch = epoch
    if args.dataset =='vimeo_triplet' :
        dataset = vimeo_triplet(split='train', path=args.data_path,crop_size=args.train_crop,resize=None)
        dataset_val = vimeo_triplet('test', args.data_path,crop_size=args.val_crop,resize=None)
    elif args.dataset =='vimeo_setuplet' :
        dataset = vimeo_setuplet(split='train', path=args.data_path,crop_size=args.train_crop,resize=None)
        dataset_val = vimeo_setuplet('test', args.data_path,crop_size=args.val_crop,resize=None)

    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=sampler)
    val_data = DataLoader(dataset_val, batch_size=1 , pin_memory=True, num_workers=1)
    step_per_epoch = train_data.__len__()
    
    if args.first_val :
        cur_psnr=evaluate(model, val_data, epoch, args)

    print('training...')    
    for epoch in range(start_epoch,args.epoch_max) :
        with tqdm(train_data ) as  tepoch:
            sampler.set_epoch(epoch)
            for  (imgs,t) in ((tepoch)):
                imgs = imgs.to(device, non_blocking=True) / 255.
                imgs, gt = imgs[:, 0:6], imgs[:, 6:]
                learning_rate = get_learning_rate(step=global_step,
                                                epoch_max=args.epoch_max,
                                                step_per_epoch=step_per_epoch,
                                                max_lr=args.max_lr,
                                                min_lr=args.min_lr,
                                                warmup_step=args.warmup_step)

                pred, loss = model.update(imgs, gt, learning_rate, timestep=t, training=True)
                psnr = peak_signal_noise_ratio(pred, gt, data_range=1.0, dim=(1,2,3)).item()

                if args.local_rank == 0:
                    #print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss:{:.4e} psnr:{}  lr:{}'.format(epoch, i, step_per_epoch, data_time_interval, train_time_interval, loss,psnr,learning_rate))
                    tepoch.set_postfix({'Epoch': epoch,'PSNR': psnr,'Loss': loss.item(),'Val_PSNR':cur_psnr,'Lr':learning_rate})
                    run.log({'epoch':epoch,'loss':loss,'PSNR':psnr ,'lr':learning_rate })
                global_step += 1
            
            if (epoch+1) % args.val_interval == 0:
                val_psnr=evaluate(model, val_data, epoch, args)
                if cur_psnr < val_psnr :
                    cur_psnr = val_psnr
                    model.save_checkpoint(epoch,global_step,val_psnr,args.local_rank)    
                
            dist.barrier()
    run.finish()

def evaluate(model, val_data, epoch, args):
    print('validation...')
    psnr = []
    for (imgs,t) in (tqdm(val_data)):
        imgs = imgs.to(device, non_blocking=True) / 255.
        imgs, gt = imgs[:, 0:6], imgs[:, 6:]
        with torch.no_grad():
            pred, _ = model.update(imgs, gt, timestep=t,training=False)
        for j in range(gt.shape[0]):
            psnr.append(-10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item()))
    psnr = np.array(psnr).mean()
    if args.local_rank == 0:
        run.log({ 'val_PSNR' : psnr , 'epoch' : epoch})
        print('epoch:{} psnr:{:.2f} dB'.format(epoch,psnr))
        #print('epoch : psnr', str(epoch), psnr)
    return psnr

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=1, type=int, help='world size')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--num_workers', default=16, type=int, help='num_workers ')
    parser.add_argument('--epoch_max', default=300, type=int, help='epoch size')
    parser.add_argument('--max_lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--min_lr', default=2e-5, type=float, help='min learning rate')
    parser.add_argument('--warmup_step', default=2000, type=int, help='warmup step')
    parser.add_argument('--val_interval', default=3, type=int, help='val interval')
    parser.add_argument('--train_crop', default=[256,256], type=Union[int, Sequence[int]], help='train crop size')
    parser.add_argument('--val_crop', default=None, type=Union[int, Sequence[int]], help='val crop size')
    parser.add_argument('--first_val', default=False, type=bool, help='first interval')
    parser.add_argument('--resume', default=None, type=str, help='resume')
    parser.add_argument('--model', default='BiFormer', type=str, help='model')
    parser.add_argument('--data_path', type=str, default='/data/dataset/vimeo_dataset/vimeo_triplet',help='data path of dataset')
    parser.add_argument('--dataset', type=str, default='vimeo_triplet',help='tpye of dataset')
    parser.add_argument('--project', type=str, default='my',help='wandb project name')
    parser.add_argument('--configs',    type=str,   default='configs/BiFormer_paper.yaml')

    args = parser.parse_args()
    cfgs = load_yml2args(args.configs)
    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    
    if args.local_rank == 0:
        wandb.login()
        log_dir = Path.cwd().absolute() / "wandb_logs" / args.model
        log_dir.mkdir(exist_ok=True, parents=True)
        sha = hashlib.sha256()
        sha.update(str(args.model).encode())
        wandb_id = sha.hexdigest()

        run = wandb.init(project=args.project,
                        id=  wandb_id, 
                        dir= log_dir,
                        job_type='train',
                        save_code=True,
                        notes='',
                        name=args.model,
                        resume='allow')

    if args.resume :
        print('resume',args.resume)
        model = Model(args.local_rank,cfgs)
        ckpt=model.load_checkpoint(args.resume,args.local_rank,training=True)
        epoch=ckpt['epoch']
        global_step=ckpt['global_step']
        cur_psnr=ckpt['psnr']
        assert args.model == ckpt['model']
        train(model, args,epoch,global_step,cur_psnr)
        #model.load_model(args.resume,args.local_rank)
        #model.save_checkpoint(300,0,35.88,args.local_rank)    
    else:
        model = Model(args.local_rank,cfgs)
        train(model, args)
        
