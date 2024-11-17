import rawpy
from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import cv2

random_seed = 42
random.seed(random_seed)

def glob_file_list(root):
    return sorted(glob.glob(os.path.join(root, '*')))


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def augment_torch(img_list, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = flip(img, 2)
        if vflip:
            img = flip(img, 1)
        return img

    return [_augment(img) for img in img_list]


class SonyDataset(Dataset):
    def __init__(self, patch_size,mode='train'):

        target_root = '/home/zdx/my_shared_data_folder/SID/long_sid2'
        source_root = '/home/zdx/my_shared_data_folder/SID/short_sid2'
        self.patch_size = patch_size
        self.source_paths = []
        self.target_paths = []
        self.mode = mode
        subfolders_LQ_origin = glob_file_list(source_root)
        subfolders_GT_origin = glob_file_list(target_root)
        subfolders_LQ = []
        subfolders_GT = []

        # 区分训练集和测试集数据 文件夹名字以 '0' 或 '2' 开头训练集
        if mode == 'train':
            for mm in range(len(subfolders_LQ_origin)):
                name = os.path.basename(subfolders_LQ_origin[mm])
                if '0' in name[0] or '2' in name[0]:
                    subfolders_LQ.append(subfolders_LQ_origin[mm])
                    subfolders_GT.append(subfolders_GT_origin[mm])
        else:
            for mm in range(len(subfolders_LQ_origin)):
                name = os.path.basename(subfolders_LQ_origin[mm])
                if '1' in name[0]:
                    subfolders_LQ.append(subfolders_LQ_origin[mm])
                    subfolders_GT.append(subfolders_GT_origin[mm])

        for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
            # subfolder_name = os.path.basename(subfolder_LQ)

            img_paths_LQ = glob_file_list(subfolder_LQ)# 获取name对应的不同曝光低光照图
            img_paths_GT_origin = glob_file_list(subfolder_GT)
            length = len(img_paths_LQ)
            img_paths_GT = []
            for mm in range(length):
                img_paths_GT.append(img_paths_GT_origin[0])# 匹配不同曝光的低光照对应的gt
            self.source_paths.extend(img_paths_LQ)
            self.target_paths.extend(img_paths_GT)
        print(f'{mode} GT:{target_root} num：{len(self.target_paths)}')
        print(f'{mode} abnomal:{source_root} num：{len(self.source_paths)}')

    def __len__(self):
        return len(self.source_paths)

    def resize_image(self, img):
        h, w, _ = img.shape
        # h_ =  h - h % 256
        h_ = h - h % 16
        # w_ = w - w % 256
        w_ = w - w % 16
        h__ = (h % 16) // 2
        w__ = (w % 16) // 2

        img = img[h__:h__ + h_, w__:w__ + w_]
        # img = img[h__:h__+h_, w__:w__+w_]
        return img

    def crop_image(self,image):
        """
        将图像裁剪为指定的目标大小。

        Parameters:
        - image: 输入的图像
        - target_size: 目标大小，格式为 (height, width)

        Returns:
        - cropped_image: 裁剪后的图像
        """

        h, w = image.shape[:2]
        th, tw = self.patch_size,self.patch_size
        if h == th and w == tw:
            return image  # 如果图像已经是目标大小，则直接返回

        # 如果图像小于目标大小，选择是否进行缩放
        if h < th or w < tw:
            scale_factor = max(th / h, tw / w)
            image = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))

        # 随机选择裁剪的起始位置
        # top = random.randint(0, max(0, image.shape[0] - th))
        # left = random.randint(0, max(0, image.shape[1] - tw))

        # 计算中心裁剪的起始位置
        top = (h - th) // 2
        left = (w - tw) // 2

        # 裁剪图像
        cropped_image = image[top:top + th, left:left + tw]
        # print(f' cropped_image.shape:{cropped_image.shape}')
        return cropped_image

    def show(self,from_im,to_im,index):
        # 创建一个包含两个子图的 Matplotlib 图形
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # 显示源图像
        axes[0].imshow(from_im)
        axes[0].set_title('Source Image')

        # 显示目标图像
        axes[1].imshow(to_im)
        axes[1].set_title('Target Image')

        # 隐藏坐标轴
        for ax in axes:
            ax.axis('off')
        # # 保存图形为图像文件（例如 PNG）
        if not os.path.exists('show/load'):
            os.makedirs('show/load')
        plt.savefig(f'show/load/{index}.png', bbox_inches='tight')
        # 显示图形
        # plt.show()

    def __getitem__(self, index):
        from_path = self.source_paths[index]
        from_im = np.load(from_path)
        from_im = from_im[:, :, [2, 1, 0]]/ 255.0# 通道的顺序从BGR转换为RGB。

        # from_im = Image.fromarray(from_im)# Numpy数组转换为图像对象

        to_path = self.target_paths[index]
        to_im = np.load(to_path)
        to_im = to_im[:, :, [2, 1, 0]]/ 255.0

        if self.mode == 'train' or self.mode == 'validation':
            from_im = self.crop_image(from_im)
            to_im = self.crop_image(to_im)
        # to_immax=np.max(to_im)
        # to_immin=np.min(to_im)

        # 数据增强
        # if self.mode == 'train':
        #     if random.randint(0, 1):
        #         to_im = flip(to_im, 2)
        #         from_im = flip(from_im, 2)

        # self.show(from_im,to_im,index)

        # to_im = (to_im + 1) * 0.5
        # from_im = (from_im + 1) * 0.5
        from_im = torch.from_numpy(from_im.astype(np.float32)).permute(2, 0, 1)
        to_im = torch.from_numpy(to_im.astype(np.float32)).permute(2, 0, 1)
        return from_im, to_im

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

class SonyDataset2(Dataset):
    def __init__(self, patch_size,mode='train'):

        target_root = '/home/zdx/my_shared_data_folder/Sony/long'
        source_root = '/home/zdx/my_shared_data_folder/Sony/short'
        self.patch_size = patch_size
        self.source_paths = []
        self.target_paths = []
        self.mode = mode
        test_fns = glob.glob(target_root + '/1*.ARW')
        test_ids = []
        for i in range(len(test_fns)):
            _, test_fn = os.path.split(test_fns[i])
            test_ids.append(int(test_fn[0:5]))

        for test_id in test_ids:
            # test the first image in each sequence
            inpath = os.path.join(source_root, '%05d_00*.ARW' % test_id)
            in_files = glob.glob(inpath)
            for k in range(len(in_files)):
                # in_path = in_files[k]
                # _, in_fn = os.path.split(in_path)
                gt_files = glob.glob(os.path.join(target_root, '%05d_00*.ARW' % test_id))
                gt_path = gt_files[0]
                self.target_paths.append(gt_path)
            self.source_paths.extend(in_files)





        print(f'{mode} GT:{target_root} num：{len(self.target_paths)}')
        print(f'{mode} abnomal:{source_root} num：{len(self.source_paths)}')

    def __len__(self):
        return len(self.source_paths)

    def resize_image(self, img):
        h, w, _ = img.shape
        # h_ =  h - h % 256
        h_ = h - h % 16
        # w_ = w - w % 256
        w_ = w - w % 16
        h__ = (h % 16) // 2
        w__ = (w % 16) // 2

        img = img[h__:h__ + h_, w__:w__ + w_]
        # img = img[h__:h__+h_, w__:w__+w_]
        return img

    def crop_image(self,image):
        """
        将图像裁剪为指定的目标大小。

        Parameters:
        - image: 输入的图像
        - target_size: 目标大小，格式为 (height, width)

        Returns:
        - cropped_image: 裁剪后的图像
        """

        h, w = image.shape[:2]
        th, tw = self.patch_size,self.patch_size
        if h == th and w == tw:
            return image  # 如果图像已经是目标大小，则直接返回

        # 如果图像小于目标大小，选择是否进行缩放
        if h < th or w < tw:
            scale_factor = max(th / h, tw / w)
            image = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))

        # 随机选择裁剪的起始位置
        # top = random.randint(0, max(0, image.shape[0] - th))
        # left = random.randint(0, max(0, image.shape[1] - tw))

        # 计算中心裁剪的起始位置
        top = (h - th) // 2
        left = (w - tw) // 2

        # 裁剪图像
        cropped_image = image[top:top + th, left:left + tw]
        # print(f' cropped_image.shape:{cropped_image.shape}')
        return cropped_image

    def show(self,from_im,to_im,index):
        # 创建一个包含两个子图的 Matplotlib 图形
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # 显示源图像
        axes[0].imshow(from_im)
        axes[0].set_title('Source Image')

        # 显示目标图像
        axes[1].imshow(to_im)
        axes[1].set_title('Target Image')

        # 隐藏坐标轴
        for ax in axes:
            ax.axis('off')
        # # 保存图形为图像文件（例如 PNG）
        if not os.path.exists('show/load'):
            os.makedirs('show/load')
        plt.savefig(f'show/load/{index}.png', bbox_inches='tight')
        # 显示图形
        # plt.show()

    def __getitem__(self, index):
        in_path = self.source_paths[index]

        gt_path = self.target_paths[index]

        _, in_fn = os.path.split(in_path)
        _, gt_fn = os.path.split(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
        # input_full = input_full[:,:512, :512, :]
        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # im = im[:1024, :1024]
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        input_full = np.minimum(input_full, 1.0)

        self.show(input_full, gt_full, index)

        from_im = torch.from_numpy(input_full.astype(np.float32)).permute(2, 0, 1)
        to_im = torch.from_numpy(gt_full.astype(np.float32)).permute(2, 0, 1)

        return from_im, to_im