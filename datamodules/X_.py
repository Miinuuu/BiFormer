import os
from typing import Any, Callable, Optional, Sequence, Union
import cv2
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import glob
import torch
from torch.utils.data import DataLoader

class X_(Dataset):
    def __init__(
        self, 
        split,
        path='/data/dataset/X4K1000FPS/', 
        crop_size: Union[int, Sequence[int]] =None,
        resize: Union[int, Sequence[int]] =None,
        max_t_step_size=32
    ):
    
        self.img_list  = []

        self.max_t_step_size = max_t_step_size
        self.split=split
        self.resize=resize

        if split == 'train':
            self.data_root =os.path.join(path, 'train')
            self.framelist = self.make_2D_dataset_X_Train(dir=self.data_root)
        elif split == 'test':
            self.data_root =os.path.join(path, 'test')
            self.framelist = self.make_2D_dataset_X_Test(dir=self.data_root,multiple=8,t_step_size=32)
        elif split =='val' :
            self.data_root =os.path.join(path, 'test')
            self.framelist = self.make_2D_dataset_X_Test(dir=self.data_root,multiple=8,t_step_size=32)
        else :
            raise RuntimeError("inappropriate dataset type")

        self.load_data()
        self.pil_transform =  ToTensor()
        self.crop_size=crop_size
    
    def frames_loader_test(self, I0I1It_Path):
        frames = []
        for path in I0I1It_Path:
            frame = cv2.imread(path)
            frames.append(frame)
        (ih, iw, c) = frame.shape
        frames = np.stack(frames, axis=0)  # (T, H, W, 3)

        if  self.split=='test' :
            if self.crop_size:  ## center crop
                H_ps,W_ps = self.crop_size[0],self.crop_size[1]
                ix = (iw - W_ps) // 2
                iy = (ih - H_ps) // 2
                frames = frames[:, iy:iy + H_ps, ix:ix + W_ps, :]

        if self.split=='val' :
            if self.crop_size:  ## center crop
                H_ps,W_ps = 512,512
                ix = (iw - W_ps) // 2
                iy = (ih - H_ps) // 2
                frames = frames[:, iy:iy + H_ps, ix:ix + W_ps, :]
        return frames
    
    def frames_loader_train(self, candidate_frames, frameRange):
        frames = []
        for frameIndex in frameRange:
            frame = cv2.imread(candidate_frames[frameIndex])
            frames.append(frame)
        (ih, iw, c) = frame.shape
        #print(frame.shape)
        frames = np.stack(frames, axis=0)  # (T, H, W, 3)
        if self.crop_size:  ## random crop
            H_ps,W_ps = self.crop_size[0],self.crop_size[1]
            ix = random.randrange(0, iw - W_ps + 1)
            iy = random.randrange(0, ih - H_ps + 1)
            frames = frames[:, iy:iy + H_ps, ix:ix + W_ps, :]
            #print(frames.shape)
        if random.random() < 0.5:  # random horizontal flip
                frames = frames[:, :, ::-1, :]
        '''if random.random() < 0.5:  # random vertical flip
                frames = frames[:, ::-1, :, :]
        if random.random() < 0.5:  # random channel flip
                frames = frames[:, :, :, ::-1]'''
            # No vertical flip
        rot = random.randint(0, 3)  # random rotate
        frames = np.rot90(frames, rot, (1, 2))
        return frames
    
    def make_2D_dataset_X_Train(self,dir):
        framesPath = []
        # Find and loop over all the clips in root `dir`.
        for scene_path in sorted(glob.glob(os.path.join(dir, '*', ''))):
            sample_paths = sorted(glob.glob(os.path.join(scene_path, '*', '')))
            for sample_path in sample_paths:
                frame65_list = []
                for frame in sorted(glob.glob(os.path.join(sample_path, '*.png'))):
                    frame65_list.append(frame)
                framesPath.append(frame65_list)

        print("The number of total training samples : {} which has 65 frames each.".format(
            len(framesPath)))  ## 4408 folders which have 65 frames each
        return framesPath

    def make_2D_dataset_X_Test(self, dir, multiple=8, t_step_size=32):
        """ make [I0,I1,It,t,scene_folder] """
        """ 1D (accumulated) """
        testPath = []
        t = np.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))
        for type_folder in sorted(glob.glob(os.path.join(dir, '*', ''))):  # [type1,type2,type3,...]
            for scene_folder in sorted(glob.glob(os.path.join(type_folder, '*', ''))):  # [scene1,scene2,..]
                frame_folder = sorted(glob.glob(scene_folder + '*.png'))  # 32 multiple, ['00000.png',...,'00032.png']
                for idx in range(0, len(frame_folder), t_step_size):  # 0,32,64,...
                    if idx == len(frame_folder) - 1:
                        break
                    for mul in range(multiple - 1):
                        I0I1It_paths = []
                        I0I1It_paths.append(frame_folder[idx])  # I0 (fix)
                        I0I1It_paths.append(frame_folder[idx + t_step_size])  # I1 (fix)
                        I0I1It_paths.append(frame_folder[idx + int((t_step_size // multiple) * (mul + 1))])  # It
                        I0I1It_paths.append(t[mul])
                        I0I1It_paths.append(scene_folder.split(os.path.join(dir, ''))[-1])  # type1/scene1
                        testPath.append(I0I1It_paths)
        print("The number of total test samples : {} which has 32 frames each.".format(len(testPath)))  ## 4408 folders which have 65 frames each
        return testPath
    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        self.meta_data = self.framelist

    def getimg_train(self, index):
        t_step_size = random.randint(2, self.max_t_step_size)
        t_list = np.linspace((1 / t_step_size), (1 - (1 / t_step_size)), (t_step_size - 1))
        
        firstFrameIdx = random.randint(0, (64 - t_step_size))
        interIdx = random.randint(1, t_step_size - 1)  # relative index, 1~self.t_step_size-1
        interFrameIdx = firstFrameIdx + interIdx  # absolute index
        
        t_value = t_list[interIdx - 1]  # [0,1]
        
        if (random.randint(0, 1)): # frame random ordering
            frameRange = [firstFrameIdx, firstFrameIdx + t_step_size, interFrameIdx]
        else:  ## temporally reversed order
            frameRange = [firstFrameIdx + t_step_size, firstFrameIdx, interFrameIdx]
            interIdx = t_step_size - interIdx  # (self.t_step_size-1) ~ 1
            t_value = 1.0 - t_value
        #print(frameRange,t_value)
        candidate_frames= self.meta_data[index]
        frames = self.frames_loader_train(candidate_frames,frameRange)  # including "np2Tensor [-1,1] normalized"
        #timestep= np.expand_dims(np.array(t_value, dtype=np.float32), 0)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)

        
        return frames[0],frames[2],frames[1],timestep
    
    def getimg_test(self, index):
        I0, I1, It, timestep, scene_name = self.meta_data[index]
        I0I1It_Path = [I0, I1, It]
        #print(I0I1It_Path,timestep)
        frames = self.frames_loader_test(I0I1It_Path)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)
        return frames[0],frames[2],frames[1],timestep
    
    def __getitem__(self, index):

        if self.split =="train" :
            img0,gt,img1,timestep = self.getimg_train(index)
        else :
            img0,gt,img1,timestep = self.getimg_test(index)

        if self.resize :
            h,w=self.resize[0],self.resize[1]
            img0=cv2.resize(img0,(w,h) )
            gt=cv2.resize(gt,(w,h) )
            img1=cv2.resize(img1,(w,h) )
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0,img1,gt ), 0),timestep
    


if __name__ == "__main__":    
    dataset_val=X_(split='train',path='/data/dataset/X4K1000FPS/',crop_size=None)
    val_data = DataLoader(dataset_val , batch_size= 1 , pin_memory = True , num_workers=1)
    for item in  val_data :       
        #print(item['batch'].shape)
        #print(item['timestep'])
        pass
