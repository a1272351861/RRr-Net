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

import data_generator as dg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def visualization(img, img_path, iteration):
    # --- Visualization for Checking --- #
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    img = img.cpu().detach().numpy()

    for i in range(img.shape[0]):
        # save name
        name = str(iteration) + '_' + str(i) + '.jpg'
        print(name)
        if img.shape[1] == 3:
            img_single = np.transpose(img[i, :, :, :], (1, 2, 0))
        else:
            img_single = np.transpose(img[i, :, :, :], (0, 1, 2))
        # print(img_single)
        img_single = np.clip(img_single, 0, 1) * 255.0
        img_single = cv2.UMat(img_single).get()
        img_single = img_single / 255.0
        plt.imsave(os.path.join(img_path, name), img_single)

def visualization2(img, img_path, iteration):
    # --- Visualization for Checking --- #
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    # img = img.cpu().numpy()
    # img = img.squeeze().cpu().detach().numpy()
    # for i in range(img.shape[0]):  ###遍历batchsize
    #     # save name
    #     name = str(iteration) + '_' + str(i) + '.jpg'
        # print(name)
        # if img.shape[1] == 3:
        #     img_single = np.transpose(img[i, :, :, :], (1, 2, 0))
        # else:
        #     img_single = np.transpose(img[i, :, :, :], (0, 1, 2))
        # print(img_single)
    # if type(img) == 'tensor':
    img = img.squeeze().permute(1, 2, 0).cpu().detach().numpy()

    inputs = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(img_path, '{:07d}_out.png'.format(iteration)), (inputs * 255).astype(np.uint8))

class DatasetRAW(object):
    def __init__(self, root, batch_size, patch_size, stride, gamma_flag = True, to_gpu=True, ftype='png'):
        self.root = root
        self.gamma_flag = gamma_flag
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.stride = stride

        # load all image files
        rawimgs=[]
        srgbimgs=[]
        
        print('Inside train_datagen \n')
        rawimgs = dg.datagenerator_allin(os.path.join(self.root, 'high'), batch_size=batch_size, patch_size=patch_size, stride=stride, ftype=ftype)
        print(rawimgs.shape[0])
        print('Outside train_datagen \n')
        rawimgs = torch.from_numpy(rawimgs.astype(np.float32))
        rawimgs = rawimgs.permute(0, 3, 1, 2)
        rawimgs = rawimgs / 65535.0
        if to_gpu:
            rawimgs = rawimgs.to(device)
        self.rawimgs = rawimgs
        
        print('Inside train_datagen \n')
        srgbimgs = dg.get_nameList(os.path.join(self.root, 'low'), batch_size=batch_size, patch_size=patch_size, stride=stride, ftype=ftype)
        print(srgbimgs.shape[0])
        print('Outside train_datagen \n')
        srgbimgs = torch.from_numpy(srgbimgs.astype(np.float32))
        srgbimgs = srgbimgs.permute(0, 3, 1, 2)
        # The max value differs according to dataset (png or tif)
        if ftype == 'png':
            srgbimgs = srgbimgs / 255.0
        elif ftype == 'tif':
            srgbimgs = srgbimgs / 65535.0
        else:
            raise ValueError('ftype is not valid.')
        if to_gpu:
            srgbimgs = srgbimgs.to(device)
        self.srgbimgs = srgbimgs

    def __getitem__(self, idx):
        # load images
        target = self.rawimgs[idx]
        img = self.srgbimgs[idx]
        return img, target

    def __len__(self):
        return len(self.rawimgs)


class DatasetRAWTest(object):
    def __init__(self, root, to_gpu=True, ftype='png'):
        self.root = root
        self.to_gpu = to_gpu
        self.ftype = ftype
        # load all image files {ndarray(w,h,3)}
        rawimgs=[]
        srgbimgs=[]
        
        print('Inside train_datagen \n')
        rawimgs = dg.get_patches(os.path.join(self.root, 'high'), ftype=ftype)
        print(len(rawimgs))
        # i= 0
        # for rawimg in rawimgs:
        #     i +=1
        #     visualization2(rawimg/255.0,'raw',i)
        print('Outside train_datagen \n')
        
        self.rawimgs = rawimgs
        
        print('Inside train_datagen \n')
        srgbimgs = dg.get_patches(os.path.join(self.root, 'low'), ftype=ftype)
        print(len(srgbimgs))
        print('Outside train_datagen \n')
        self.srgbimgs = srgbimgs

    def __getitem__(self, idx):
        # load images
        target = self.rawimgs[idx]
        target = torch.from_numpy(target.astype(np.float32))
        target = target.permute(2, 0, 1)
        # target = target / 65535.0
        target = target / 255.0
        if self.to_gpu:
            target = target.to(device)

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
        if self.to_gpu:
            img = img.to(device)

        return img, target

    def __len__(self):
        return len(self.rawimgs)