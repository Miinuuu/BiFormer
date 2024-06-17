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
class UCF101_test_triplet(Dataset):
    def __init__(self, 
        split,
        path='/data/dataset/ucf101/', 
                crop_size: Union[int, Sequence[int]] =None,
                resize: Union[int, Sequence[int]] =None,

    ):
        self.path = path
        self.pil_transform =  ToTensor()
        self.img_list = []
        self.name = 'UCF101'
        self.dir_seq = os.listdir(self.path)
        self.dir_seq.sort()
        self.resize=resize
        self.crop_size=crop_size
    def __getitem__(self, idx):
        img_path = self.dir_seq[idx]

        img0 =  cv2.imread(os.path.join(self.path, img_path ,'frame_00.png'))
        gt =  cv2.imread(os.path.join(self.path, img_path ,'frame_01_gt.png'))
        img1 =  cv2.imread(os.path.join(self.path, img_path ,'frame_02.png'))

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

    def __len__(self):
        return len(self.dir_seq)