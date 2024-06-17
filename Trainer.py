import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from model.loss import *
from imageio import mimsave,imsave
from model import *
from model.loss import *
#from warplayer import *
class Model:
    def __init__(self, local_rank,cfgs):

        self.BiFormer = BiFormer(**cfgs.transformer_cfgs)
        self.Upsampler = Upsampler(**cfgs.Upsampler_cfgs)
        self.SynNet = SynNet()


        self.name = 'BiFormer'
        self.device()
        # train
        self.CensusLoss=CensusLoss()
        params = [
            {'params': self.BiFormer.parameters()},
            {'params': self.Upsampler.parameters()}, 
            {'params': self.SynNet.parameters()}, 
            ] # model2에 다른 학습률 적용

        self.optimG = AdamW(params, lr=2e-4, weight_decay=1e-4)
        self.lap = LapLoss()
        if local_rank != -1:
            self.BiFormer = DDP(self.BiFormer, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
            self.Upsampler = DDP(self.Upsampler, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
            self.SynNet = DDP(self.SynNet, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)

    def train(self):
        self.BiFormer.train()
        self.Upsampler.train()
        self.SynNet.train()

    def eval(self):
        self.BiFormer.eval()
        self.Upsampler.eval()
        self.SynNet.eval()

    def device(self):
        self.BiFormer.to(torch.device("cuda"))
        self.SynNet.to(torch.device("cuda"))
        self.Upsampler.to(torch.device("cuda"))

    def load_model(self, ckpt_path):
        checkpoint = torch.load(ckpt_path,map_location='cpu')
        self.BiFormer.load_state_dict(checkpoint['BiFormer_state_dict'], strict=True)
        self.Upsampler.load_state_dict(checkpoint['Upsampler_state_dict'], strict=True)
        self.SynNet.load_state_dict(checkpoint['SynNet_state_dict'], strict=True)
            
    def load_checkpoint(self ,name=None,rank=0,training=False):
        
        def convert(param):
            return {
            k.replace("module.", ""): v
                #for k, v in param['model_state_dict'].items()
                for k, v in param.items()
                if "module." in k and 'attn_mask' not in k and 'HW' not in k
            }
        epoch=0
        global_step=0
        if rank <= 0 :
            if name is None:
                name = self.name
            print('---load_checkpoint---')
            print('resume : ',name)

            checkpoint = (torch.load(f'ckpt/{name}.pkl'))
            if training :
                self.BiFormer.load_state_dict(checkpoint['BiFormer_state_dict'],strict=True)
                self.Upsampler.load_state_dict(checkpoint['Upsampler_state_dict'],strict=True)
                self.SynNet.load_state_dict(checkpoint['SynNet_state_dict'],strict=True)
            else:
                #self.net.load_state_dict(convert(checkpoint),strict=True)
                self.BiFormer.load_state_dict(convert(checkpoint['BiFormer_state_dict']),strict=True)
                self.Upsampler.load_state_dict(convert(checkpoint['Upsampler_state_dict']),strict=True)
                self.SynNet.load_state_dict(convert(checkpoint['SynNet_state_dict']),strict=True)
            
            self.optimG.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            global_step = checkpoint['global_step']
            psnr = checkpoint['psnr']
            name = checkpoint['name']
            l1_loss=torch.nn.L1Loss()
            print('model:',name)
            print('psnr:',psnr)
            print('epoch:',epoch)
            print('global_step:',global_step)

        return {'Model':name ,'epoch':epoch,'global_step':global_step,'psnr':psnr}

    def save_checkpoint(self,epoch,global_step,psnr,rank=0):
        if rank == 0:
            checkpoint={
                'name':self.name,
                'psnr' : psnr,
                'epoch': epoch,
                'global_step': global_step,
                'BiFormer_state_dict': self.BiFormer.state_dict(),
                'Upsampler_state_dict': self.Upsampler.state_dict(),
                'SynNet_state_dict': self.SynNet.state_dict(),
                'optimizer_state_dict': self.optimG.state_dict(),
            }
            print('---save_checkpoint---')
            print('Model:',self.name)
            print('psnr:',psnr)
            print('epoch:',epoch)
            print('global_step:',global_step)     

            torch.save(checkpoint,f'ckpt/{self.name}'+'_'+str(epoch)+'_'+str(psnr)[0:5]+'.pkl')

    def warp(self, x, flo):
            """
            warp an image/tensor (im2) back to im1, according to the optical flow

            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow

            """
            B, C, H, W = x.size()
            # mesh grid
            xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
            yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)

            grid = torch.cat((xx, yy), 1).float()

            if x.is_cuda:
                grid = grid.to(x.device)

            vgrid = torch.autograd.Variable(grid) + flo

            # scale grid to [-1,1]
            vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
            vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

            vgrid = vgrid.permute(0, 2, 3, 1)
            output = nn.functional.grid_sample(x, vgrid, align_corners=True)
            mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
            mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

            mask = mask.masked_fill_(mask < 0.999, 0)
            mask = mask.masked_fill_(mask > 0, 1)

            return output * mask
    def update(self, imgs, gt, learning_rate=0,timestep=0.5, training=True):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()

        if training:
            #print('imgsshape',imgs.shape)
            img1,img3=imgs[:,0:3],imgs[:,3:6]
            img1_prev,img3_prev= imgs[:,0:3],imgs[:,3:6]
            flow_fw =self.BiFormer(img1_prev, img3_prev)
            _, _, H_ori, W_ori = img1_prev.shape

            for iter in reversed(range(1,3)):
                H_ = H_ori // (2**iter)
                W_ = W_ori // (2**iter)
                img1_prev = F.interpolate(img1_prev, (H_,W_), mode='bilinear')
                img3_prev = F.interpolate(img3_prev, (H_,W_), mode='bilinear')
                
                _,_,H_c,W_c = flow_fw.shape
                flow_fw = F.interpolate(flow_fw, (H_, W_), mode='bilinear')
                flow_fw[:,0,:,:] *= W_ / float(W_c)
                flow_fw[:,1,:,:] *= H_ / float(H_c)
                
                flow_fw = self.Upsampler(img1_prev, img3_prev, flow_fw)

            _,_,H_c,W_c = flow_fw.shape
            flow_fw = F.interpolate(flow_fw, (H_ori, W_ori), mode='bilinear')
            flow_fw[:,0,:,:] *= W_ori / float(W_c)
            flow_fw[:,1,:,:] *= H_ori / float(H_c)
            # Based on linear motion assumption
            flow_bw = flow_fw * (-1)
            pred = self.SynNet(img1, img3, flow_bw, flow_fw)

            
            l_photo = charbonnier_loss( gt, self.warp(img1, flow_bw))
            +charbonnier_loss(gt, self.warp(img3, flow_fw))
            +self.CensusLoss(gt,self.warp(img1, flow_bw))
            +self.CensusLoss(gt,self.warp(img3, flow_fw)) 

            L_syn = charbonnier_loss(pred,gt)
            +self.CensusLoss(gt,pred)

            l_pho=l_photo+L_syn
            
            self.optimG.zero_grad()
            l_pho.backward()
            self.optimG.step()
            return pred, l_pho
        else: 
            with torch.no_grad():
                img1_prev,img3_prev= imgs[:,0:3],imgs[:,3:6]
                flow_fw =self.BiFormer(img1_prev, img3_prev)
                _, _, H_ori, W_ori = img1_prev.shape

                for iter in reversed(range(1,3)):
                    H_ = H_ori // (2**iter)
                    W_ = W_ori // (2**iter)
                    img1_prev = F.interpolate(img1_prev, (H_,W_), mode='bilinear')
                    img3_prev = F.interpolate(img3_prev, (H_,W_), mode='bilinear')
                    
                    _,_,H_c,W_c = flow_fw.shape
                    flow_fw = F.interpolate(flow_fw, (H_, W_), mode='bilinear')
                    flow_fw[:,0,:,:] *= W_ / float(W_c)
                    flow_fw[:,1,:,:] *= H_ / float(H_c)
                    
                    flow_fw = self.Upsampler(img1_prev, img3_prev, flow_fw)

                _,_,H_c,W_c = flow_fw.shape
                flow_fw = F.interpolate(flow_fw, (H_ori, W_ori), mode='bilinear')
                flow_fw[:,0,:,:] *= W_ori / float(W_c)
                flow_fw[:,1,:,:] *= H_ori / float(H_c)
                # Based on linear motion assumption
                flow_bw = flow_fw * (-1)
                pred = self.SynNet(imgs[:,0:3], imgs[:,3:6], flow_bw, flow_fw)
                
                return pred, 0
