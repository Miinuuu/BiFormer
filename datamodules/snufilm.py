### Triplet test set ###
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

class snufilm(Dataset):
    def  __init__(self, 
        split,
        path='/data/dataset/snufilm/', 
        mode='extreme', 
        crop_size: Union[int, Sequence[int]] =None,
        resize: Union[int, Sequence[int]] =None,

    ):
        self.path = path
        test_fn = os.path.join(path, 'eval_modes/test-%s.txt' % mode)
        self.name = 'SNU_FILM'+mode
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()
        self.testlist = [v.split(' ') for v in self.testlist]
        
        self.load_data()
        self.crop_size=crop_size
        self.resize=resize

        print("[%s] Test dataset has %d triplets" % (mode, len(self.testlist)))
        self.pil_transform =  ToTensor()

    def load_data(self):
        self.meta_data = self.testlist

    def crop(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1
    
    def getimg(self, index):
        imgpaths = self.meta_data[index]

        imgpath0 =os.path.join(self.path,imgpaths[0] )
        imgpath1 =os.path.join(self.path,imgpaths[1] )
        imgpath2 =os.path.join(self.path,imgpaths[2] )

        img0 = cv2.imread(imgpath0)
        gt = cv2.imread(imgpath1)
        img1 = cv2.imread(imgpath2)

        return img0, gt, img1

    def __getitem__(self, index):
        img0, gt, img1 = self.getimg(index)

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
    
    def __len__(self):
        return len(self.meta_data)


if __name__ == "__main__":    

    dataset_val=snufilm(split=None,path='/data/dataset/snufilm/')
    val_data = DataLoader(dataset_val , batch_size= 1 , pin_memory = True , num_workers=1)
    for item in  val_data :
        print(item.shape)

