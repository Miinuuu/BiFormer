import cv2
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from typing import List, Optional, Sequence, Union
from torch.utils.data import DataLoader
import flow_vis

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class vimeo_triplet_flow_bi(Dataset):
    def __init__(self, 
                split, 
                path,
                crop_size: Union[int, Sequence[int]] =None,
                resize=None):
        self.split = split
        self.data_root = path
        self.resize=resize
        self.crop_size = crop_size

        self.image_root = os.path.join(self.data_root, 'sequences')
        self.flow_root = os.path.join(self.data_root, 'sequences_flows_bi')
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()                                                    
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        if self.split == 'train':
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist

    def aug(self, img0, gt, img1, flow_gt, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        flow_gt = flow_gt[x:x+h, y:y+w, :]

        return img0, gt, img1,flow_gt

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        flowpath = os.path.join(self.flow_root, self.meta_data[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']
        flowpaths = [flowpath + '/flo31.npy', flowpath + '/flo13.npy']
        
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        timestep=0.5


        flow21 = np.load(flowpaths[0])
        flow23 = np.load(flowpaths[1])
        flow_gt = np.concatenate([flow21, flow23], axis=0).transpose(1, 2, 0)
        return img0, gt, img1,flow_gt,timestep
            
    def __getitem__(self, index):        
        img0, gt, img1,flow_gt,timestep= self.getimg(index)
                
        if 'train' in self.split:
            if self.crop_size :
                h,w = self.crop_size
                img0, gt, img1 ,flow_gt= self.aug(img0, gt, img1,flow_gt, h, w)            

            if random.uniform(0, 1) < 0.5:# vertical flip
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
                flow_gt = flow_gt[::-1]
                flow_gt = np.concatenate((flow_gt[:, :, 0:1], -flow_gt[:, :, 1:2], flow_gt[:, :, 2:3], -flow_gt[:, :, 3:4]), 2)
            if random.uniform(0, 1) < 0.5:# horizontal flip
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
                flow_gt = flow_gt[:, ::-1]
                flow_gt = np.concatenate((-flow_gt[:, :, 0:1], flow_gt[:, :, 1:2], -flow_gt[:, :, 2:3], flow_gt[:, :, 3:4]), 2)
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                flow_gt = np.concatenate((flow_gt[:, :, 2:4], flow_gt[:, :, 0:2]), 2)
                timestep = 1 - timestep

            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
                flow_gt=np.rot90(flow_gt,k=1,axes=(0, 1))
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gt = cv2.rotate(gt, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
                flow_gt=np.rot90(flow_gt,k=2,axes=(0, 1))

            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                flow_gt=np.rot90(flow_gt,k=-1,axes=(0, 1))

        #if self.resize :
        #    h,w=self.resize[0],self.resize[1]
        #    img0=cv2.resize(img0,(w,h) )
        #    gt=cv2.resize(gt,(w,h) )
        #    img1=cv2.resize(img1,(w,h) )
        #    flow_gt = torch.from_numpy(flow_gt).float().permute(2, 0, 1)

        flow_gt = torch.from_numpy(flow_gt.copy()).permute(2, 0, 1)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)

        return torch.cat((img0,img1,gt,flow_gt), 0), timestep

    
    #torch.cat((img0, img1, gt), 0),flow_gt,timestep

if __name__ == "__main__":    
    dataset_val=vimeo_triplet_flow_bi(split='train',path='/data/dataset/vimeo_dataset/vimeo_triplet',crop_size=None)
    val_data = DataLoader(dataset_val , batch_size= 1 , pin_memory = True , num_workers=1)
    for i,item in  enumerate(val_data) :       
        #print(item['img0'].shape)
        
        #print(item['flow_gt'].shape)
        #print(item[''])
        
        flow=item['flow_gt'][0].permute(1, 2, 0).detach().cpu().numpy()
        print(flow.min(),flow.max())
        flow13 = flow_vis.flow_to_color(flow[:,:,0:2], convert_to_bgr=False)    
        flow31 = flow_vis.flow_to_color(flow[:,:,2:4], convert_to_bgr=False)   
        path = './img/'+str(i)+'/'
        if not os.path.exists(path):
                        os.makedirs(path)
        cv2.imwrite('./img/'+str(i)+'/'+'0.png',(item['img0'][0].detach().cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)) 
        cv2.imwrite('./img/'+str(i)+'/'+'1.png',(item['img1'][0].detach().cpu().numpy().transpose(1, 2, 0)).astype(np.uint8))
        cv2.imwrite('./img/'+str(i)+'/'+'gt.png',(item['gt'][0].detach().cpu().numpy().transpose(1, 2, 0)).astype(np.uint8))
        cv2.imwrite('./img/'+str(i)+'/'+'flow13.png',(flow13))
        cv2.imwrite('./img/'+str(i)+'/'+'flow31.png',(flow31))
    
