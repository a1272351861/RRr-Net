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

import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import sys
sys.path.append("./utils")
sys.path.append("../")

import data_generator as dg
import torch.utils.data as data

import numpy as np
from PIL import Image
from glob import glob
import random
random.seed(1143)
from torchvision.transforms import Compose, ToTensor, Normalize, ConvertImageDtype
import torchvision.transforms.functional as TF
from data_generator import gen_patches,get_image,get_image_size,get_patches224
import torchvision.transforms as T


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def visualization( img_path,img:torch.Tensor, iteration=''):
    # --- Visualization for Checking --- #
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    # img = img.cpu().numpy()
    if isinstance(img, torch.Tensor) :
        img = img.cpu().detach().numpy()
        # img = img.squeeze().cpu().detach().numpy()
    if len(img.shape) == 3:

        if img.shape[0] == 3 :
            img_single = np.transpose(img[:, :, :], (1, 2, 0))
        elif img.shape[0] == 1:
            img_single = img.squeeze()
        else:
            img_single = np.transpose(img[:, :, :], (0, 1, 2))
        name = 'iteration' + '_' + str(iteration) + '.jpg'
        # img_single = np.clip(img_single, 0, 1) * 255.0
        # img_single = cv2.UMat(img_single).get()
        # img_single = img_single / 255.0
        ### plt.imsave()不用*255
        plt.imsave(os.path.join(img_path, name), img_single)
    elif len(img.shape) == 4:
        for i in range(img.shape[0]):  ###遍历batchsize
            # save name
            name = str(iteration) + '_' + str(i) + '.jpg'
            # print(name)
            if img.shape[1] == 3 :
                img_single = np.transpose(img[i, :, :, :], (1, 2, 0))
            elif img.shape[1] == 1:
                img_single = np.transpose(img[i, 0, :, :], (0, 1))
            else:
                img_single = np.transpose(img[i, :, :, :], (0, 1, 2))
            # print(img_single)
            img_single = np.clip(img_single, 0, 1) * 255.0
            img_single = cv2.UMat(img_single).get()
            img_single = img_single / 255.0
            plt.imsave(os.path.join(img_path, name), img_single)
    elif len(img.shape) == 2:
        # 绘制灰度特征图
        name = 'iteration' + '_' + str(iteration) + '.jpg'
        plt.imsave(os.path.join(img_path,name),img,  cmap='gray')


def visualization2( img_path, inputs:torch.Tensor,targets=None, outputs=None,iteration=1):
    # --- Visualization for Checking --- #
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    # 保存一张inputs
    if not isinstance(targets, torch.Tensor) and not isinstance(outputs, torch.Tensor) :
        inputs = inputs.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        cv2.imwrite(os.path.join(img_path, '{:07d}_out.png'.format(iteration)), (inputs * 255).astype(np.uint8))
        # print('one')
        return
    # 保存三张
    if isinstance(inputs, torch.Tensor) and isinstance(targets, torch.Tensor) and isinstance(outputs, torch.Tensor):
        inputs = inputs.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        targets = targets.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        outputs = outputs.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        images_to_concat = [inputs, targets, outputs]
        combined_image = np.concatenate(images_to_concat, axis=1)
        combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
        # inputs = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # print('three')
        cv2.imwrite(os.path.join(img_path, '{:07d}_out.png'.format(iteration)), (combined_image * 255).astype(np.uint8))
    else:
        raise ValueError("Invalid input types.")

class DatasetRAW_exp(object):
    def __init__(self, root, batch_size=64, patch_size=128, stride=52, gamma_flag=True, to_gpu=True, ftype='jpg',mode='train'):
        self.root = root
        self.gamma_flag = gamma_flag
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.stride = stride
        self.ftype = ftype
        self.mode = mode
        device = to_gpu
        # torch.cuda.set_device(to_gpu)
        # print(device)
        # load all image files
        img_high = []
        img_low = []
        if self.mode == 'train' or self.mode == 'val':
            self.gt_path = os.path.join(self.root, 'GT')
        elif self.mode == 'test':
            self.gt_path = os.path.join(self.root,  'expert_' + 'c' + '_testing_set')
        print('GTpath: ',self.gt_path,' \n')


        # print('Inside train_datagen \n')
        self.img_lows_name = dg.get_nameList(os.path.join(self.root, 'abnormal'), name=True, ftype=ftype)

    def FLIP_aug(self, low, high):
        if random.random() > 0.5:
            low = cv2.flip(low, 0)
            high = cv2.flip(high, 0)

        if random.random() > 0.5:
            low = cv2.flip(low, 1)
            high = cv2.flip(high, 1)

        return low, high

    def __getitem__(self, idx):
        # load images
        ####获取名字.jpg
        img_id = self.img_lows_name[idx]
        a = img_id.rfind('_')
        ####获取名字
        img_id_gt = img_id[:a]

        img_lows_path = os.path.join(self.root, 'abnormal',img_id)
        img_gt_path = os.path.join(self.gt_path, img_id_gt + '.%s' % self.ftype)

        if self.mode == 'train' or self.mode == 'val':
            img_low = get_image_size(img_lows_path)
            img_gt = get_image_size(img_gt_path)
            img_low, img_gt = self.FLIP_aug(img_low, img_gt)
        else:
            img_low = get_image_size(img_lows_path,size=512)
            img_gt = get_image_size(img_gt_path,size=512)

            if img_low.shape[0] != img_gt.shape[0]:
                img_low = cv2.rotate(img_low, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img_low = torch.from_numpy(np.array(img_low, dtype='float32'))
        img_gt = torch.from_numpy(np.array(img_gt, dtype='float32'))
        img_low = img_low / 255.0
        img_gt = img_gt / 255.0

        # visualization(img_low,'show/train_low',idx)
        # visualization(img_gt,'show/train_gt',idx)

        img = img_low.permute(2, 0, 1)
        target = img_gt.permute(2, 0, 1)

        return img, target

    def __len__(self):
        return len(self.img_lows_name)

class DatasetRAW(object):
    def __init__(self, root, batch_size, patch_size, stride, gamma_flag = True, to_gpu=True, ftype='png'):
        self.root = root
        self.gamma_flag = gamma_flag
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.stride = stride
        device = to_gpu
        # torch.cuda.set_device(to_gpu)
        print(device)
        # load all image files
        img_high=[]
        img_low=[]
        
        # print('Inside train_datagen')
        # 获取名字.jpg list
        img_high = dg.get_nameList(os.path.join(self.root, 'GT'), ftype=ftype)
        print('GT:',len(img_high))
        self.img_highs = img_high
        
        # print('Inside train_datagen ')
        img_low = dg.get_nameList(os.path.join(self.root, 'abnormal'), ftype=ftype)
        print('abnormal:',len(img_low))

        self.img_lows = img_low

    def __getitem__(self, idx):
        # load images

        img_high = gen_patches(self.img_highs[idx], patch_size=self.patch_size, stride=self.stride)
        img_lows = gen_patches(self.img_lows[idx], patch_size=self.patch_size, stride=self.stride)

        img_high = np.concatenate(img_high)  # , dtype='uint8')
        img_lows = np.concatenate(img_lows)  # , dtype='uint8')

        discard_n = len(img_high) - len(img_high) // self.batch_size * self.batch_size;# 计算了要删除的数据数量 discard_n

        img_high = np.delete(img_high, range(discard_n), axis=0)  #这是为了保证 data 数组的长度是 batch_size 的整数倍
        img_lows = np.delete(img_lows, range(discard_n), axis=0)

        img_high = torch.from_numpy(np.array(img_high, dtype='float32'))
        img_lows = torch.from_numpy(np.array(img_lows, dtype='float32'))

        img_high = img_high / 255.0
        img_lows = img_lows / 255.0

        # visualization(img_high,'show/train_img_high',idx)
        # visualization(img_lows,'show/train_img_lows',idx)
        target = img_high.permute(2, 0, 1)
        img = img_lows.permute(2, 0, 1)

        return img, target
        ###得到b,60,3,patch,patch 需要改，不要数据增强
    def __len__(self):
        return len(self.img_highs)

##
class Dataset_sketch(object):
    def __init__(self, root, batch_size, patch_size, stride, gamma_flag=True, train=True, ftype='jpg'):
        self.root = root
        self.gamma_flag = gamma_flag
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.stride = stride
        self.train = train
        # self.stride = stride
        # device = to_gpu
        # torch.cuda.set_device(to_gpu)
        # print(device)


        # print('Inside train_datagen')
        # 获取名字.jpg list
        img_high = dg.get_nameList(os.path.join(self.root, 'GT'), ftype=ftype)
        print(os.path.join(self.root, 'GT'),':', len(img_high))
        self.img_highs = img_high

        # print('Inside train_datagen ')
        img_low = dg.get_nameList(os.path.join(self.root, 'abnormal'), ftype=ftype)
        print(os.path.join(self.root, 'abnormal'),':', len(img_low))

        self.img_lows = img_low

    def __getitem__(self, idx):
        # load images
        img_high = dg.get_image(self.img_highs[idx])
        img_low = dg.get_image(self.img_lows[idx])

        height = img_high.shape[0]
        width = img_high.shape[1]
        img_sketch = self.get_sketch(img_high)#60,1,b,b


        img_high = torch.from_numpy(np.array(img_high, dtype='float32'))
        img_low = torch.from_numpy(np.array(img_low, dtype='float32'))
        # img_sketch = torch.from_numpy(np.array(img_sketch, dtype='float32'))

        img_high = img_high.permute(2, 0, 1)
        img_low = img_low.permute(2, 0, 1)
        # img_sketch = img_sketch.permute(2, 0, 1)

        img_high = img_high/255.0
        img_low = img_low/255.0



        img_sketch[img_sketch == 255] = 1
        img_sketch = cv2.resize(img_sketch, (width, height))
        # visualization(sketch,'show/train_img_sketch')
        img_sketch = torch.from_numpy(img_sketch).permute(2, 0, 1)
        img_sketch = img_sketch[0:1, :, :]
        img_sketch = img_sketch.long()

        # if self.train:
        #     if random.randint(0, 1):
        #         img_high=T.RandomRotation(degrees=(20, 60))(img_high)
        #         img_low=T.RandomRotation(degrees=(20, 60))(img_low)
        #         img_sketch=T.RandomRotation(degrees=(20, 60))(img_sketch)
                # img_low = T.RandomRotation(90)
                # img_sketch = T.RandomRotation(90)


        visualization(img_high,'show/train_img_high',idx)
        visualization(img_low,'show/train_img_low',idx)
        visualization(img_sketch,'show/train_img_sketch',idx)
        # print(idx,img_sketch.shape)
        return img_low, img_high,img_sketch

    def flip(self,x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    def get_sketch(self,img):
        result = []

        # img =np.array(img, dtype='float32')
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)###灰度图
        sketch = cv2.GaussianBlur(im_gray, (3, 3), 0)

        v = np.median(sketch)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        sketch = sketch.astype(np.uint8)
        sketch = cv2.Canny(sketch, lower, upper)  ###边缘检测图

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        sketch = cv2.dilate(sketch, kernel)

        sketch = np.expand_dims(sketch, axis=-1)
        sketch = np.concatenate([sketch, sketch, sketch], axis=-1)
        # assert len(np.unique(sketch)) == 2



        return sketch

    def __len__(self):
        return len(self.img_highs)

# class Datasetsketch_allin(object):
#     def __init__(self, root, batch_size, patch_size, stride, gamma_flag=True, to_gpu=True, ftype='jpg'):
#         self.root = root
#         self.gamma_flag = gamma_flag
#         self.batch_size = batch_size
#         self.patch_size = patch_size
#         self.stride = stride
#         device = to_gpu
#         # torch.cuda.set_device(to_gpu)
#         print(device)
#         # load all image files
#         img_high = []
#         img_low = []
#
#         print(os.path.join(self.root,'GT'))
#         img_high = dg.datagenerator2(os.path.join(self.root, 'GT'), batch_size=batch_size, patch_size=patch_size,
#                                     stride=stride)
#         print(img_high.shape[0])
#
#         img_sketch = dg.get_sketch(img_high)
#         self.img_sketch = img_sketch
#
#         img_high = torch.from_numpy(img_high.astype(np.float32))
#
#         img_high = img_high.permute(0, 3, 1, 2)
#         if ftype == 'raw':
#             img_high = img_high / 65535.0
#         else:
#             img_high = img_high / 255.0
#         # if to_gpu:
#         # img_high = img_high.cuda()
#         #     # img_high = img_high.to(device)
#         self.img_highs = img_high
#
#         print(os.path.join(self.root,'abnormal'))
#         img_low = dg.datagenerator2(os.path.join(self.root, 'abnormal'), batch_size=batch_size, patch_size=patch_size,
#                                    stride=stride)
#         print(img_low.shape[0])
#
#         img_low = torch.from_numpy(img_low.astype(np.float32))
#         img_low = img_low.permute(0, 3, 1, 2)
#         # The max value differs according to dataset (png or tif)
#         if ftype == 'png':
#             img_low = img_low / 255.0
#         elif ftype == 'tif':
#             img_low = img_low / 65535.0
#         else:
#             # raise ValueError('ftype is not valid.')
#             img_low = img_low / 255.0
#         # if to_gpu:
#         # img_low = img_low.cuda()
#         #     # img_low = img_low.to(device)
#         self.img_lows = img_low
#
#
#
#     def __getitem__(self, idx):
#         # load images
#         target = self.img_highs[idx]
#         img = self.img_lows[idx]
#         img_sketch = self.img_sketch[idx]
#         return img, target,img_sketch
#
#     def __len__(self):
#         return len(self.img_highs)

class DatasetRAW_allin(object):
    def __init__(self, root, batch_size, patch_size, stride, gamma_flag=True, to_gpu=True, ftype='jpg'):
        self.root = root
        self.gamma_flag = gamma_flag
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.stride = stride
        device = to_gpu
        # torch.cuda.set_device(to_gpu)
        # print(device)
        # load all image files
        img_high = []
        img_low = []

        # print(os.path.join(self.root,'GT'))
        img_high = dg.datagenerator_allin(os.path.join(self.root, 'GT'), batch_size=batch_size, patch_size=patch_size,
                                          stride=stride)
        # print(img_high.shape[0])
        print('Train: ',os.path.join(self.root,'GT'),img_high.shape[0])

        img_high = torch.from_numpy(img_high.astype(np.float32))
        img_high = img_high.permute(0, 3, 1, 2)
        if ftype == 'raw':
            img_high = img_high / 65535.0
        else:
            img_high = img_high / 255.0
        # if to_gpu:
        # img_high = img_high.cuda()
        #     # img_high = img_high.to(device)
        self.img_highs = img_high

        img_low = dg.datagenerator_allin(os.path.join(self.root, 'abnormal'), batch_size=batch_size, patch_size=patch_size,
                                         stride=stride)
        print('Train: ',os.path.join(self.root,'abnormal'),img_low.shape[0])

        img_low = torch.from_numpy(img_low.astype(np.float32))
        img_low = img_low.permute(0, 3, 1, 2)
        # The max value differs according to dataset (png or tif)
        if ftype == 'png':
            img_low = img_low / 255.0
        elif ftype == 'tif':
            img_low = img_low / 65535.0
        else:
            # raise ValueError('ftype is not valid.')
            img_low = img_low / 255.0
        # if to_gpu:
        # img_low = img_low.cuda()
        #     # img_low = img_low.to(device)
        self.img_lows = img_low



    def __getitem__(self, idx):
        # load images
        target = self.img_highs[idx]
        img = self.img_lows[idx]
        # visualization2('show/train/gt', target,iteration=idx)
        # visualization2('show/train/low', img,iteration=idx)
        return img, target

    def __len__(self):
        return len(self.img_highs)

####先搞进一个大数组
class DatasetRAWTest(object):
    def __init__(self, root, to_gpu=True, ftype='jpg'):
        self.root = root
        self.to_gpu = to_gpu
        self.ftype = ftype
        # load all image files {ndarray(w,h,3)}
        rawimgs=[]
        srgbimgs=[]
        # rawimgs = dg.datagenerator_allin(os.path.join(self.root, 'GT'), batch_size=1, patch_size=224, stride=52)
        if ftype == 'exdark':
            rawimgs = dg.get_patches(self.root)
        else:
            rawimgs = dg.get_patches(os.path.join(self.root,'GT'))
        print('test: ', os.path.join(self.root, 'GT'), len(rawimgs))

        # i= 0
        # for rawimg in rawimgs:
        #     i +=1
        #     visualization2(rawimg/255.0,'raw',i)
        # print('Outside train_datagen \n')
        
        self.rawimgs = rawimgs
        # print(os.path.join(self.root,'abnormal'))
        # srgbimgs = dg.get_patches224(os.path.join(self.root, 'abnormal'))
        # srgbimgs = dg.datagenerator_allin(os.path.join(self.root, 'abnormal'),batch_size=1, patch_size=224, stride=52)

        if ftype == 'exdark':
            srgbimgs = None
        else:
            srgbimgs = dg.get_patches(os.path.join(self.root,'abnormal'))
        print('test: ',os.path.join(self.root,'abnormal'),len(srgbimgs))

        # print(len(srgbimgs))
        # print('Outside train_datagen \n')
        self.srgbimgs = srgbimgs

    def __getitem__(self, idx):
        # load images
        target = self.rawimgs[idx]
        target = torch.from_numpy(target.astype(np.float32))
        target = target.permute(2, 0, 1)
        if self.ftype == 'raw':
            target = target / 65535.0
        else:
            target = target / 255.0
        if self.to_gpu:
            target = target.cuda()

        img = self.srgbimgs[idx]
        if img.shape[0] != self.rawimgs[idx].shape[0]:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = torch.from_numpy(img.astype(np.float32))
        img = img.permute(2, 0, 1)
        # The max value differs according to dataset (png or tif)
        if self.ftype == 'png':
            # img = img
            img = img/255.0
        elif self.ftype == 'tif':
            # img = img
            img = img/65535.0
        else:
            img = img / 255.0
        if self.to_gpu:
            img = img.cuda()

        # visualization2('show/test/gt', target,iteration=idx)
        visualization('show/test/gt2',target,iteration=idx)
        visualization('show/test/img',img,iteration=idx)
        # visualization2('show/test/low', img,iteration=idx)
        return img, target

    def __len__(self):
        return len(self.rawimgs)

# calculate params
def calculate_params(model):
    if not isinstance(model, list):
        params = list(model.parameters())
    else:
        params = model
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("总参数数量和：" + str(k))

##############################################################################
def populate_train_list(images_path, mode='train'):
    # print(images_path)
    image_list_lowlight = glob(images_path + '*.png')
    train_list = image_list_lowlight
    if mode == 'train':
        random.shuffle(train_list)

    return train_list

class lowlight_loader_new(data.Dataset):

    def __init__(self, images_path, mode='train'):
        self.train_list = populate_train_list(images_path, mode)
        self.mode = mode
        self.data_list = self.train_list
        print("Total examples:", len(self.train_list))

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]
        ps = 256  # Training Patch Size
        if self.mode == 'train':
            data_lowlight = Image.open(data_lowlight_path).convert('RGB')
            data_highlight = Image.open(data_lowlight_path.replace('low', 'high')).convert('RGB')
            w, h = data_lowlight.size
            data_lowlight = TF.to_tensor(data_lowlight)
            data_highlight = TF.to_tensor(data_highlight)
            hh, ww = data_highlight.shape[1], data_highlight.shape[2]

            rr = random.randint(0, hh - ps)
            cc = random.randint(0, ww - ps)
            aug = random.randint(0, 8)

            # Crop patch
            data_lowlight = data_lowlight[:, rr:rr + ps, cc:cc + ps]
            data_highlight = data_highlight[:, rr:rr + ps, cc:cc + ps]

            # Data Augmentations
            if aug == 1:
                data_lowlight = data_lowlight.flip(1)
                data_highlight = data_highlight.flip(1)
            elif aug == 2:
                data_lowlight = data_lowlight.flip(2)
                data_highlight = data_highlight.flip(2)
            elif aug == 3:
                data_lowlight = torch.rot90(data_lowlight, dims=(1, 2))
                data_highlight = torch.rot90(data_highlight, dims=(1, 2))
            elif aug == 4:
                data_lowlight = torch.rot90(data_lowlight, dims=(1, 2), k=2)
                data_highlight = torch.rot90(data_highlight, dims=(1, 2), k=2)
            elif aug == 5:
                data_lowlight = torch.rot90(data_lowlight, dims=(1, 2), k=3)
                data_highlight = torch.rot90(data_highlight, dims=(1, 2), k=3)
            elif aug == 6:
                data_lowlight = torch.rot90(data_lowlight.flip(1), dims=(1, 2))
                data_highlight = torch.rot90(data_highlight.flip(1), dims=(1, 2))
            elif aug == 7:
                data_lowlight = torch.rot90(data_lowlight.flip(2), dims=(1, 2))
                data_highlight = torch.rot90(data_highlight.flip(2), dims=(1, 2))

            filename = os.path.splitext(os.path.split(data_lowlight_path)[-1])[0]

            return data_lowlight, data_highlight, filename

        elif self.mode == 'val':
            data_lowlight = Image.open(data_lowlight_path).convert('RGB')
            data_highlight = Image.open(data_lowlight_path.replace('low', 'high')).convert('RGB')
            # Validate on center crop
            if ps is not None:
                data_lowlight = TF.center_crop(data_lowlight, (ps, ps))
                data_highlight = TF.center_crop(data_highlight, (ps, ps))

            data_lowlight = TF.to_tensor(data_lowlight)
            data_highlight = TF.to_tensor(data_highlight)

            filename = os.path.splitext(os.path.split(data_lowlight_path)[-1])[0]

            return data_lowlight, data_highlight, filename

        elif self.mode == 'test':
            data_lowlight = Image.open(data_lowlight_path).convert('RGB')
            data_highlight = Image.open(data_lowlight_path.replace('low', 'high')).convert('RGB')

            data_lowlight = TF.to_tensor(data_lowlight)
            data_highlight = TF.to_tensor(data_highlight)

            filename = os.path.splitext(os.path.split(data_lowlight_path)[-1])[0]
            # print(filename)
            return data_lowlight, data_highlight, filename

    def __len__(self):
        return len(self.data_list)