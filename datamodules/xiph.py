import os
from typing import Any, Callable, Optional, Sequence, Union
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


class xiph(Dataset):
    def __init__(
            self, 
            split,
            path='/data/dataset/Xiph/', 
            t_frame=99, 
            crop_size: Union[int, Sequence[int]] =None,
            resize: Union[int, Sequence[int]] =None

        ):
        self.path = path
        self.img_list = []
        self.crop_size=crop_size
        self.resize=resize
        for idx, dataset in enumerate(os.listdir(os.path.join(path,'test_4k'))):
#            path+'test_4k')):
            # if idx != 13:
            #     print(idx, dataset)
            #     continue
            # print('*', idx, dataset)
            base_dir = path + 'test_4k/' + dataset

            #for i in range(1, t_frame-2):
            for i in range(2, t_frame,2):
                im1_path = base_dir + '/' + '%03d' % (i -1) + '.' + 'png'
                im2_path = base_dir + '/' + '%03d' % (i ) + '.' + 'png'
                im3_path = base_dir + '/' + '%03d' % (i + 1) + '.' + 'png'
                #print(im3_path)
                self.img_list.append([im1_path, im2_path, im3_path])

        self.pil_transform =  ToTensor()
        self.name = 'Xiph4K_full'

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        
        #print(self.img_list[idx])
        img0 =  cv2.imread(self.img_list[idx][0])
        gt =  cv2.imread(self.img_list[idx][1])
        img1 =  cv2.imread(self.img_list[idx][2])

        if self.crop_size != None: #center crop
            #print('selfcrop_size',self.crop_size)
            #print(img1.shape)
            h,w = self.crop_size
            h=h//2
            w=w//2
            img0=img0[h:-h, w:-w, :]
            gt=gt[h:-h, w:-w, :]
            img1=img1[h:-h, w:-w, :]
            #print(img1.shape)

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

    dataset_val=xiph(split=None,path='/data/dataset/Xiph/',t_frame=100)
    val_data = DataLoader(dataset_val ,batch_size= 1 , pin_memory = True , num_workers=4)
    for item in  val_data :
        pass
        print(item.shape)

