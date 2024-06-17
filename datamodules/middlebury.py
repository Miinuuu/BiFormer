import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union
import cv2
import numpy as np
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader


class middlebury(Dataset):
    def __init__(
            self, 
            split,
            path='/data/dataset/middlebury/', 
            crop_size: Union[int, Sequence[int]] =None,
            resize: Union[int, Sequence[int]] =None,

        ):
        self.crop_size=crop_size
        self.resize=resize

        self.path = path
        self.img_list = []
        for idx, dataset in enumerate(os.listdir(os.path.join(path,'other-data'))):

            base_dir = path + 'test_4k/' + dataset
            base_dir = os.path.join(path,'other-data',dataset)
            base_dir_gt = os.path.join(path,'other-gt-interp',dataset)


            im1_path = base_dir + '/'+'frame10.png'
            im2_path = base_dir_gt + '/' + 'frame10i11.png'
            im3_path = base_dir + '/'+'frame11.png'
                #print(im3_path)
            self.img_list.append([im1_path, im2_path, im3_path])

        self.pil_transform =  ToTensor()
        self.name = 'Xiph4K_full'

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        
        img0 =  cv2.imread(self.img_list[idx][0])
        gt =  cv2.imread(self.img_list[idx][1])
        img1 =  cv2.imread(self.img_list[idx][2])

        if self.crop_size != None:
            h,w = self.crop_size
            img0, gt, img1 = self.crop(img0, gt, img1,h,w)
            
        if self.resize :
            #print('resize')
            h,w=self.resize[0],self.resize[1]
            img0=cv2.resize(img0,(w,h) )
            gt=cv2.resize(gt,(w,h) )
            img1=cv2.resize(img1,(w,h) )
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        timestep = torch.tensor(0.5).reshape(1, 1, 1)

        return torch.cat((img0,img1,gt ), 0),timestep
    

if __name__ == "__main__":    

    dataset_val=middlebury(split=None,path='/data/dataset/middlebury/')
    val_data = DataLoader(dataset_val ,batch_size= 1 , pin_memory = True , num_workers=4)
    for item in  val_data :
        #pass
        print(item.shape)

