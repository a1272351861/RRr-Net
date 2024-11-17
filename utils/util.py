"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from skimage.segmentation import mark_boundaries


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    sketch = [item[2] for item in batch]
    data = torch.stack(data, dim=0)
    target = torch.stack(target, dim=0)
    sketch = torch.stack(sketch, dim=0)
    return [data, target,sketch]


def poolfeat(input, prob, sp_h=2, sp_w=2,device='cuda:0'):
    # torch.cuda.set_device(gpu_id)
    def feat_prob_sum(feat_sum, prob_sum, shift_feat):
        feat_sum += shift_feat[:, :-1, :, :]
        prob_sum += shift_feat[:, -1:, :, :]
        return feat_sum, prob_sum

    b, _, h, w = input.shape

    h_shift_unit = 1
    w_shift_unit = 1
    p2d = (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit)
    feat_ = torch.cat([input, torch.ones([b, 1, h, w]).to(device)], dim=1)  # b* (n+1) *h*w
    # feat_ = input
    # feat_ = torch.cat([input, torch.ones([b, 1, h, w]).cuda()], dim=1)  # b* (n+1) *h*w
    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 0, 1), kernel_size=(sp_h, sp_w),stride=(sp_h, sp_w)) # b * (n+1) * h* w
    send_to_top_left =  F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, 2 * w_shift_unit:]
    feat_sum = send_to_top_left[:, :-1, :, :].clone()
    prob_sum = send_to_top_left[:, -1:, :, :].clone()

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 1, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum,prob_sum,top )

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 2, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top_right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 3, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    left = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, h_shift_unit:-h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 4, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    center = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, center)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 5, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    right = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, h_shift_unit:-h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 6, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_left = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, :-2 * h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 7, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 8, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_right)


    pooled_feat = feat_sum / (prob_sum + 1e-8)#b,8,16,16

    return pooled_feat

def upfeat(input, prob, up_h=2, up_w=2):
    # input b*n*H*W  downsampled
    # prob b*9*h*w
    b, c, h, w = input.shape

    h_shift = 1
    w_shift = 1

    p2d = (w_shift, w_shift, h_shift, h_shift)
    feat_pd = F.pad(input, p2d, mode='constant', value=0)#b,8,18,18

    gt_frm_top_left = F.interpolate(feat_pd[:, :, :-2 * h_shift, :-2 * w_shift], size=(h * up_h, w * up_w),mode='nearest')#b,8,128,128
    feat_sum = gt_frm_top_left * prob.narrow(1,0,1)#b,8,128,128

    top = F.interpolate(feat_pd[:, :, :-2 * h_shift, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += top * prob.narrow(1, 1, 1)

    top_right = F.interpolate(feat_pd[:, :, :-2 * h_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += top_right * prob.narrow(1,2,1)

    left = F.interpolate(feat_pd[:, :, h_shift:-w_shift, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += left * prob.narrow(1, 3, 1)

    center = F.interpolate(input, (h * up_h, w * up_w), mode='nearest')
    feat_sum += center * prob.narrow(1, 4, 1)

    right = F.interpolate(feat_pd[:, :, h_shift:-w_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += right * prob.narrow(1, 5, 1)

    bottom_left = F.interpolate(feat_pd[:, :, 2 * h_shift:, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom_left * prob.narrow(1, 6, 1)

    bottom = F.interpolate(feat_pd[:, :, 2 * h_shift:, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom * prob.narrow(1, 7, 1)

    bottom_right =  F.interpolate(feat_pd[:, :, 2 * h_shift:, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom_right * prob.narrow(1, 8, 1) #b,8,128,128

    return feat_sum

def update_spixl_map(spixl_map_idx_in, assig_map_in):
    assig_map = assig_map_in.clone()

    b,_,h,w = assig_map.shape
    _, _, id_h, id_w = spixl_map_idx_in.shape

    if (id_h == h) and (id_w == w):
        spixl_map_idx = spixl_map_idx_in
    else:
        spixl_map_idx = F.interpolate(spixl_map_idx_in, size=(h,w), mode='nearest')

    assig_max,_ = torch.max(assig_map, dim=1, keepdim= True)
    assignment_ = torch.where(assig_map == assig_max, torch.ones(assig_map.shape).cuda(),torch.zeros(assig_map.shape).cuda())
    new_spixl_map_ = spixl_map_idx * assignment_ # winner take all
    new_spixl_map = torch.sum(new_spixl_map_,dim=1,keepdim=True).type(torch.int)

    return new_spixl_map


def get_spixel_image(given_img, spix_index):

    if not isinstance(given_img, np.ndarray):
        given_img_np = given_img.detach().cpu().numpy().transpose(1,2,0)
    else: # for cvt lab to rgb case
        given_img_np = given_img

    if not isinstance(spix_index, np.ndarray):
        spix_index_np = spix_index.detach().cpu().numpy()#.transpose(0,1)
    else:
        spix_index_np = spix_index


    # h, w = spix_index_np.shape
    # given_img_np = cv2.resize(given_img_np_, dsize=(w, h), interpolation=cv2.INTER_CUBIC)

    # if b_enforce_connect:
    #     spix_index_np = spix_index_np.astype(np.int64)
    #     segment_size = (given_img_np_.shape[0] * given_img_np_.shape[1]) / (int(n_spixels) * 1.0)
    #     min_size = int(0.06 * segment_size)
    #     max_size =  int(3 * segment_size)
    #     spix_index_np = enforce_connectivity(spix_index_np[None, :, :], min_size, max_size)[0]
    spixel_bd_image = mark_boundaries(given_img_np, spix_index_np.astype(int), color = (0,1,1)) #cyna
    return spixel_bd_image.astype(np.float32)#.transpose(2,0,1) #

def spixlIdx(n_spixl_h, n_spixl_w):
    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor = shift9pos(spix_values)

    torch_spix_idx_tensor = torch.from_numpy(
        np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float).cuda()

    return torch_spix_idx_tensor

def shift9pos(input, h_shift_unit=1,  w_shift_unit=1):
    # input should be padding as (c, 1+ height+1, 1+width+1)
    input_pd = np.pad(input, ((h_shift_unit, h_shift_unit), (w_shift_unit, w_shift_unit)), mode='edge')
    input_pd = np.expand_dims(input_pd, axis=0)

    # assign to ...
    top     = input_pd[:, :-2 * h_shift_unit,          w_shift_unit:-w_shift_unit]
    bottom  = input_pd[:, 2 * h_shift_unit:,           w_shift_unit:-w_shift_unit]
    left    = input_pd[:, h_shift_unit:-h_shift_unit,  :-2 * w_shift_unit]
    right   = input_pd[:, h_shift_unit:-h_shift_unit,  2 * w_shift_unit:]

    center = input_pd[:,h_shift_unit:-h_shift_unit,w_shift_unit:-w_shift_unit]

    bottom_right    = input_pd[:, 2 * h_shift_unit:,   2 * w_shift_unit:]
    bottom_left     = input_pd[:, 2 * h_shift_unit:,   :-2 * w_shift_unit]
    top_right       = input_pd[:, :-2 * h_shift_unit,  2 * w_shift_unit:]
    top_left        = input_pd[:, :-2 * h_shift_unit,  :-2 * w_shift_unit]

    shift_tensor = np.concatenate([     top_left,    top,      top_right,
                                        left,        center,      right,
                                        bottom_left, bottom,    bottom_right], axis=0)
    return shift_tensor

def get_kmap_from_prob(prob, grid_size):
    prob_pad = F.pad(prob, (grid_size, grid_size, grid_size, grid_size))#b,9,144,144
    prob_sp = F.pixel_unshuffle(prob_pad, grid_size)#b,576,18,18
    prob_sp = prob_sp.view(prob_sp.size()[0], prob_pad.size()[1], grid_size ** 2, prob_sp.size()[2], prob_sp.size()[3])#b,576*18*18

    prob_sp = torch.cat((prob_sp[:, 8, :, :-2, :-2], prob_sp[:, 7, :, :-2, 1:-1], prob_sp[:, 6, :, :-2, 2:],
                        prob_sp[:, 5, :, 1:-1, :-2], prob_sp[:, 4, :, 1:-1, 1:-1], prob_sp[:, 3, :, 1:-1, 2:],
                        prob_sp[:, 2, :, 2:, :-2], prob_sp[:, 1, :, 2:, 1:-1], prob_sp[:, 0, :, 2:, 2:]), 1)
    # prob_sp = torch.cat(( prob_sp[:, 3, :, 1:-1, 2:],
    #                     prob_sp[:, 2, :, 2:, :-2], prob_sp[:, 1, :, 2:, 1:-1], prob_sp[:, 0, :, 2:, 2:]), 1)
    prob_sp = prob_sp.view(prob_sp.size()[0], 3, 3, grid_size, grid_size, prob_sp.size()[-2], prob_sp.size()[-1])
    prob_sp = prob_sp.transpose(2, 3).contiguous().view(prob_sp.size()[0], -1, prob_sp.size()[-2], prob_sp.size()[-1])
    index = prob_sp.topk(prob_sp.size()[1] - 1, dim=1, largest=False)[1]
    src = torch.zeros(index.size()).to(prob.device)
    prob_sp = prob_sp.scatter(1, index, src)
    prob_sp = prob_sp.view(prob_sp.size()[0], prob_sp.size()[1], -1)
    prob_sp = F.fold(prob_sp, prob_pad.size()[2:], kernel_size=3 * grid_size, stride=grid_size)
    prob_sp = prob_sp[:, :, grid_size:-grid_size, grid_size:-grid_size]
    kmap = STEFunction.apply(prob_sp, 0)#b,1,128,128
    return kmap

import cv2
import torch
import torch.nn.functional as F
import lpips as lpips_o
from skimage.metrics import structural_similarity as ssim_o
from skimage.metrics import peak_signal_noise_ratio as psnr_o



def get_kmap_from_prob2(prob, grid_size):
    b, c, h, w = prob.shape
    # # 对概率分布进行像素不重排
    # prob_sp = F.pixel_unshuffle(prob, grid_size)  # b,192,16,16
    #
    # # 重新排列张量的形状以获得K-Map的输入形式
    # prob_sp = prob_sp.view(prob_sp.size()[0], -1, h,w)  # b,3,64,18,18
    #
    # # 合并通道以创建K-Map输入
    # kmap = torch.sum(prob_sp, dim=1, keepdim=True)  # b,1,64,18,18
    kmap = (torch.rand((b, 1, h, w)) < 0.05).float()

    return kmap

def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1

class Measure():
    def __init__(self, net='alex', use_gpu=False):
        self.device = 'cuda' if use_gpu else 'cpu'
        self.model = lpips_o.LPIPS(net=net)
        self.model.to(self.device)

    def measure(self, imgA, imgB):
        return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim, self.lpips]]


    def lpips(self,imgAbatch, imgBbatch, device='cuda'):
        for i in range(imgAbatch.shape[0]):
            imgA = imgAbatch[i, :, :, :].permute(1, 2, 0)
            imgB = imgBbatch[i, :, :, :].permute(1, 2, 0)
            imgA = (imgA.clone().detach().cpu().numpy()) * 255.0
            imgB = (imgB.clone().detach().cpu().numpy()) * 255.0
            imgA = np.clip(imgA, a_min=0.0, a_max=255.0).astype(np.uint8)
            imgB = np.clip(imgB, a_min=0.0, a_max=255.0).astype(np.uint8)
            tA = t(imgA).to(self.device)
            tB = t(imgB).to(self.device)
            dist01 = self.model.forward(tA, tB).item()
        return dist01
    # def ssim(self, imgA, imgB, gray_scale=True):
    #     if gray_scale:
    #         score, diff = ssim_o(cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY), full=True, multichannel=True)
    #     # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
    #     else:
    #         score, diff = ssim_o(imgA, imgB, full=True, multichannel=True)
    #     return score
    #
    # def psnr(self, imgA, imgB):
    #     psnr_val = psnr_o(imgA, imgB)
    #     return psnr_val


def lpips(imgAbatch, imgBbatch, device='cuda'):
    for i in range(imgAbatch.shape[0]):
        imgA = imgAbatch[i, :, :, :].permute(1, 2, 0)
        imgB = imgBbatch[i, :, :, :].permute(1, 2, 0)
        imgA = (imgA.clone().detach().cpu().numpy()) * 255.0
        imgB = (imgB.clone().detach().cpu().numpy()) * 255.0
        imgA = np.clip(imgA, a_min=0.0, a_max=255.0).astype(np.uint8)
        imgB = np.clip(imgB, a_min=0.0, a_max=255.0).astype(np.uint8)
        tA = t(imgA).to(device)
        tB = t(imgB).to(device)
        model = lpips_o.LPIPS(net='alex')
        model.to(device)
        dist01 = model.forward(tA, tB).item()
    return dist01


# def calc_psnr(img1, img2):
#     mse = torch.mean(((img1).floor() - (img2).floor()) ** 2, dim=[1, 2, 3])
#     return torch.mean(20 * torch.log10(1.0/ torch.sqrt(mse)))
def calc_psnr(img1, img2):

    mse = torch.mean(((img1 * 255).floor() - (img2 * 255).floor()) ** 2, dim=[1, 2, 3])
    # print("mse:", mse)
    PSNR = torch.mean(20 * torch.log10(255.0 / torch.sqrt(mse)))
    # print("psnr:", PSNR)

    try:
        assert not torch.isnan(img1).any(), "NaN values found in mse"
        assert not torch.isinf(img2).any(), "Inf values found in mse"
        assert not torch.isinf(mse).any(), "Inf values found in mse"
        assert not torch.isinf(PSNR).any(), "Inf values found in PSNR"
    except AssertionError as e:
        print("Assertion failed:", e)
    return PSNR
# def calc_psnr(img1, img2):
#     mse = torch.mean((img1.floor() - img2.floor()) ** 2,dim=[1,2,3])
#     return 10 * torch.log10((1.0 / mse))

class PSNR(nn.Module):
    def __init__(self, max_val=0):
        super().__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return 0

        return 10 * torch.log10((1.0 / mse))

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh):
        return (input > thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output),None
    
class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x,thresh):
        xout = STEFunction.apply(x,thresh)
        return xout