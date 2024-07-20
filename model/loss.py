import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gauss_kernel(channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1]))

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels)
        
    def forward(self, input, target):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel.to(device=input.device), max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel.to(device=target.device), max_levels=self.max_levels)
        #pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        #pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

class Ternary(nn.Module):
    def __init__(self, device):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape(
            (patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to(device)

    def transform(self, img):
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, img0, img1):
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)
    
def charbonnier_loss(x, y, eps=1e-3):
    diff = x - y
    loss = torch.sqrt(diff * diff + eps * eps)
    return loss.mean()
from torch.nn.functional import conv2d, pad, l1_loss


class CensusLoss(nn.Module):
    '''
    original implementation from
    https://github.com/dvlab-research/VFIformer/blob/main/models/losses.py
    '''
    def __init__(self, patch_size=7):
        super(CensusLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        out_channels = patch_size * patch_size
        self.patch_size = patch_size
        self.padding = int(patch_size / 2)
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to(self.device)

    def census_transform(self, x):
        patches = conv2d(x, self.w, padding=self.padding, bias=None)
        transf = patches - x
        transf = transf / torch.sqrt(0.81 + transf ** 2)
        return transf

    def rgb2gray(self, x):
        r, g, b = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
        grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return grayscale.unsqueeze(1)

    def hamming_distance(self, x0, x1):
        dist = (x0 - x1) ** 2
        dist_norm = dist / (0.1 + dist)
        dist_norm = torch.mean(dist_norm, 1, keepdim=True)
        return dist_norm

    def valid_mask(self, x, padding):
        b, _, h, w = x.shape
        valid_regions = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).float().to(self.device)
        valid_mask = pad(valid_regions, [padding, padding, padding, padding])
        return valid_mask

    def forward(self, x0, x1):
        _x0 = self.census_transform(self.rgb2gray(x0))
        _x1 = self.census_transform(self.rgb2gray(x1))
        valid_mask = self.valid_mask(_x0, 1)
        loss = self.hamming_distance(_x0, _x1) * valid_mask
        return loss.mean()
