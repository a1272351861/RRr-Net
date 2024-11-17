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

import glob
import os
from glob import glob
import cv2
import numpy as np
import torch
import random
from PIL import Image
import torchvision.transforms.functional as TF
random.seed(1143)
aug_times = 1



def data_aug(img, mode=0):
    # random_integer = random.randint(0, 7)
    # mode = random_integer
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name,patch_size=48,stride=48):

    # read image
    try:
        img = cv2.imread(file_name,cv2.IMREAD_UNCHANGED)  # -1 => any depth, in this case 16
    except AssertionError as e:
        print("Assertion failed(图片空数据):", e)

    img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=np.float32)
    h, w, cc = img.shape
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w, cc = img.shape
    patches = []
    # extract patches 一张变多张
    for i in range(0, h-patch_size+1, stride):
        for j in range(0, w-patch_size+1, stride):
            x = img[i:i+patch_size, j:j+patch_size,:]
            # data aug
            for k in range(0, aug_times):
                x_aug = data_aug(x, mode=0)
                patches.append(x_aug)###数据增强
            
    return patches

####无数据增强
def gen_patches2(file_name, patch_size=48, stride=48):
    # read image
    try:
        img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)  # -1 => any depth, in this case 16
    except AssertionError as e:
        print("Assertion failed(图片空数据):", e)

    img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=np.float32)
    h, w, cc = img.shape
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w, cc = img.shape
    patches = []
    # extract patches
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            x = img[i:i + patch_size, j:j + patch_size, :]
            # data aug
            # for k in range(0, aug_times):
            #     x_aug = data_aug(x, mode=2)
            patches.append(x)

    return patches

def get_sketch(imgs):
    one = True
    for img in imgs:
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

        height = img.shape[0]
        width = img.shape[1]
        sketch[sketch == 255] = 1
        sketch = cv2.resize(sketch, (width, height))
        sketch = torch.from_numpy(sketch).permute(2, 0, 1)
        sketch = sketch[0:1, :, :]
        sketch = sketch.long()
        if one :
            one = False
            result = sketch.unsqueeze(0)
        else:
            sketch=sketch.unsqueeze(0)
            result = torch.cat((result, sketch), 0)
    return result


####all in
def datagenerator_allin(data_dir, batch_size=128, patch_size=48, stride=48):
    
    file_list = sorted(glob(data_dir+'/*'))  # get name list of all files
    # file_list = sorted(glob.glob(data_dir+'/*.{}'.format(ftype)))  # get name list of all .png files
    # initialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patch = gen_patches2(file_list[i], patch_size=patch_size, stride=stride)
        data.append(patch)
    data = np.concatenate(data)#, dtype='uint8' all,patch,patch,3
#    data = data.reshape((data.shape[0]*data.shape[1],data.shape[2],data.shape[3],1))
    discard_n = len(data)-len(data)//batch_size*batch_size;
    data = np.delete(data,range(discard_n),axis = 0)#计算了要删除的数据数量 discard_n，这是为了保证 data 数组的长度是 batch_size 的整数倍

    return data


# 获取名字.jpg list
def get_nameList(data_dir, name=False, ftype='jpg'):
    if name:
        file_list = [os.path.basename(f) for f in glob(data_dir + '/*')]
        # file_list = [os.path.basename(f) for f in glob(os.path.join(data_dir, '*.{}'.format(ftype)))]
        file_list.sort()
    else:
        file_list = sorted(glob(data_dir + '/*'))  # get name list of all files

    return file_list

def get_image(file_name):
    # read image
    img = cv2.imread(file_name,cv2.IMREAD_UNCHANGED)  # -1 => any depth, in this case 16
    img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=np.float32)
    h, w, _ = img.shape
    # h_ =  h - h % 256
    h_ = h - h % 16
    # w_ = w - w % 256
    w_ = w - w % 16
    h__ = (h % 16) // 2
    w__ = (w % 16) // 2

    img = img[h__:h__+h_, w__:w__+w_]
    # img = img[h__:h__+h_, w__:w__+w_]
    return img

def get_image_size(file_name, size=128):
    # read image
    img = cv2.imread(file_name,cv2.IMREAD_UNCHANGED)  # -1 => any depth, in this case 16
    img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=np.float32)
    img = cv2.resize(img,(size,size))

    # img = img[h__:h__+h_, w__:w__+w_]
    return img

def get_patches(data_dir):
    
    file_list = sorted(glob(data_dir + '/*'))  # get name list of all .png files
    # file_list = sorted(glob.glob(data_dir+'/*.{}'.format(ftype)))  # get name list of all .png files
    # initialize
    data = []
    # generate patches no data Augmentation
    for i in range(len(file_list)):
        patch = get_image(file_list[i])
        data.append(patch)
    # print('^_^-training data finished-^_^')
    return data

##中心裁剪 224
def get_patches224(data_dir):
    file_list = sorted(glob(data_dir + '/*'))  # get name list of all .png files
    # file_list = sorted(glob.glob(data_dir+'/*.{}'.format(ftype)))  # get name list of all .png files
    # initialize
    data = []
    # generate patches no data Augmentation
    for i in range(len(file_list)):
        img = cv2.imread(file_list[i], cv2.IMREAD_UNCHANGED)  # -1 => any depth, in this case 16
        img_pil = Image.fromarray(np.uint8(img))  # 转换为PIL图像
        cropped_img_pil = TF.center_crop(img_pil, (224, 224))  # 中心裁剪
        img = np.array(cropped_img_pil)   # 转换回NumPy数组并归一化
        patch = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=np.float32)

        data.append(patch)
    # print('^_^-training data finished-^_^')
    return data