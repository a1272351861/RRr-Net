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
import argparse
import numpy as np

import torch
import torch.nn.functional as F

from modules2.isp_model_main import UNet_isp,UNet_reverse
from modules2.model import UNet

from utils.dataset_lol import DatasetRAWTest,visualization,visualization2, calculate_params,DatasetRAW_exp
from utils.util import calc_psnr, get_kmap_from_prob, Measure
from pytorch_msssim import ssim

###task = 'lol' 'exp' 'raw' ‘exdark’
task = 'exdark'
nokmap = '3' #1:no kmap 2:no kmap,kmap*out
rawreverse = True
target_sample = True
finetune = True

parser = argparse.ArgumentParser(description='Full fixed samples')
if task == 'exp':
    parser.add_argument('--data-dir', default='/home/zdx/my_shared_data_folder/Exposure_Correction', type=str, help='folder of training and validation images')
    testName = 'test'
    print('exp')
elif task == 'lol':
    parser.add_argument('--data-dir', default='/home/zdx/my_shared_data_folder/LOLdataset', type=str, help='folder of training and validation images')
    testName = 'eval_v2'
    print('lol')
elif task == 'raw':
    parser.add_argument('--data-dir', default='/home/zdx/paper_experiments/content-aware-metadata-main/SamsungNX2000', type=str, help='folder of training and validation images')
    testName = 'test'
    print('raw')
elif task == 'exdark':
    parser.add_argument('--data-dir', default='/home/zdx/my_shared_data_folder/Exdark/', type=str, help='folder of training and validation images')
    testName = 'lie_imgs'
    print('exdark')

parser.add_argument(
    '--file-type', default='jpg', type=str, help='image file type (png or tif)')
# parser.add_argument(
#     '--checkpoint-dir', default='/home/zdx/paper_experiments/content-aware-metadata-main/models/checkpiont/samsung/', type=str, help='folder of checkpoint')
parser.add_argument('--checkpoint-dir', default=\
'/home/zdx/paper_experiments/content-aware-metadata-main/models/lol-noaugmentation_r&finetune+isp_faster*3_nofreeze_originalsampler_finetuner-ft8-skip_reconstruct-7in-8ft_lr0.001_e120_b32_ft8_', type=str, help='folder of checkpoint')
parser.add_argument(
    '--num-iters', type=int, default=10, help='number of iterations')
parser.add_argument(
    '--patch-size', type=int, default=128, help='patch size')
parser.add_argument(
    '--init-features', type=int, default=8, help='init_features of UNet')
parser.add_argument(
    '--k', type=float, default=1.5625, help='percentage of samples to pick')
parser.add_argument(
    '--batch-size', type=int, default=1, help='batch size (DO NOT CHANGE)')
parser.add_argument(
    '--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument(
    '--extra', type=str, default='', help='extra identifier for save folder name')
if task == 'exp':
    parser.add_argument(
        '--test-filename', type=str, default='test', help='test-filename(eval15)')
else:
    parser.add_argument(
        '--test-filename', type=str, default=testName, help='test-filename(eval15)')

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

grid_size = args.patch_size ** 2 * args.k / 100
if np.sqrt(grid_size) != int(np.sqrt(grid_size)):
    print('Warning: superpixel grid seeds may not match the percentage of samples.')
grid_size = args.patch_size // int(np.sqrt(grid_size))

savefoldername=('k' + str(args.k)
    + '_lr' + str(args.lr)
    + '_i' + str(args.num_iters)
    + '_b' + str(args.batch_size)
    + '_ft' + str(args.init_features)
    + args.extra
)

root = os.path.join('./outputs/', savefoldername)
if not os.path.exists(root):
    os.makedirs(root)


# image_datasets = {x: DatasetRAWTest(os.path.join(args.data_dir,x), ftype=args.file_type)
#                   for x in [args.test_filename]}

if task == 'exp':
    image_datasets = {x: DatasetRAW_exp(os.path.join(args.data_dir, x), ftype=args.file_type, mode='test')
                      for x in [args.test_filename]}
elif task == 'lol' or task == 'raw' or task == 'exdark':

    image_datasets = {x: DatasetRAWTest(os.path.join(args.data_dir,x), ftype=task)
                      for x in [args.test_filename]}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,shuffle=False, num_workers=0)
              for x in [args.test_filename]}

dataset_sizes = {x: len(image_datasets[x]) for x in [args.test_filename]}


# if rawreverse:
#     ##返回2个输出
#     sampler = UNet_reverse(in_channels=6, out_channels=9, init_features=args.init_features, sigmoid=False)
# else:
#     sampler = UNet_isp(in_channels=6, out_channels=9, init_features=args.init_features, sigmoid=False)

if target_sample:
    sampler = UNet(in_channels=6, out_channels=9, init_features=args.init_features, sigmoid=False)
else:
    sampler = UNet_reverse(in_channels=3, out_channels=4, init_features=args.init_features, sigmoid=False)

if finetune:
    # fine_tune = UNet_isp(in_channels=3, out_channels=3, init_features=4)
    fine_tune = UNet_isp(in_channels=6, out_channels=3, init_features=8)

    fine_tune.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'best_finetune.pt')))
    fine_tune = fine_tune.to(device)
    fine_tune.eval()


# # if args.checkpoint_dir is not None:
# #     # model.load_state_dict(torch.load(config.pretrain_dir))
# #     sampler_checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best_sampler.pt'))
# #     my_dic = sampler.state_dict()
# #     # 判断预训练模型中网络的模块是否修改后的网络中也存在，并且shape相同，如果相同则取出
# #     pretrained_dict = {k: v for k, v in sampler_checkpoint.items() if k in my_dic and (v.shape == sampler_checkpoint[k].shape)}
# #     my_dic.update(pretrained_dict)
# #     sampler.load_state_dict(my_dic)
# #     print('the stupid sampler_model with pretrain')
# #
sampler.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'best_sampler.pt')))
sampler = sampler.to(device)
sampler.eval()



if nokmap == '1':
    reconstructor = UNet_isp(in_channels=6, out_channels=3, init_features=args.init_features)
elif nokmap == '2':
    reconstructor = UNet_isp(in_channels=6, out_channels=3, init_features=args.init_features)
else:
    reconstructor = UNet_isp(in_channels=7, out_channels=3, init_features=args.init_features)


# if args.checkpoint_dir is not None:
#     # model.load_state_dict(torch.load(config.pretrain_dir))
#     reconstructor_checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best_reconstructor.pt'))
#     my_dic = reconstructor.state_dict()
#     # 判断预训练模型中网络的模块是否修改后的网络中也存在，并且shape相同，如果相同则取出
#     pretrained_dict = {k: v for k, v in reconstructor_checkpoint.items() if k in my_dic and (v.shape == reconstructor_checkpoint[k].shape)}
#     my_dic.update(pretrained_dict)
#     reconstructor.load_state_dict(my_dic)
#     print('the stupid reconstructor_model with pretrain')
#
reconstructor.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'best_reconstructor.pt')))
reconstructor = reconstructor.to(device)
reconstructor.eval()

params = list(sampler.parameters()) + list(reconstructor.parameters()) + list(fine_tune.parameters())
calculate_params(params)



avg_psnr = 0
avg_ssim = 0
avg_lpips = 0
measure = Measure()

for i, (inputs, targets) in enumerate(dataloaders[args.test_filename]):
    # inputs(b,3,h,w)
    visualization2('show/test/targets',targets, iteration=i)
    # forward: sampler
    inputs, targets = inputs.to(device), targets.to(device)####cuda1变all0

    if not rawreverse:
        # outputs = sampler(torch.cat((inputs, targets), 1))
        outputs,outputs3c = sampler(inputs)

    else:
        # outputs = sampler(torch.cat((inputs, targets), 1))
        outputs= sampler(torch.cat((inputs, targets), 1))
        # print("2 return")


    prob = F.softmax(outputs, 1)
    # # sampling process
    kmap = get_kmap_from_prob(prob, grid_size).to(device)
    if nokmap == '1':
        inputs2 = torch.cat([inputs, kmap * outputs3c], dim=1)  ####nokmap
    elif nokmap == '2':
        # inputs2 = torch.cat([outputs3c, outputs], dim=1)  ####nokmap
        inputs2 = torch.cat([ kmap *outputs3c, targets], dim=1)
        # inputs2 = torch.cat([inputs, outputs], dim=1)  ####nokmap
        # inputs2 = inputs  ####nokmap
    else:
        # inputs2 = torch.cat([ inputs,kmap * outputs3c, kmap], dim=1)  # (b,3+3+1,h,w)
        inputs2 = torch.cat([ inputs,kmap * targets, kmap], dim=1)  # (b,3+3+1,h,w)


    # online_sampler = copy.deepcopy(reconstructor)
    # online_sampler.eval()
    # optimizer = optim.Adam(online_sampler.parameters(), lr=args.lr)
    #
    # for _ in range(args.num_iters):
    #     optimizer.zero_grad()
    #     outputs = online_sampler(inputs2.detach()) #(b,3,h,w)
    #     loss = ((outputs - targets).abs()).mean()
    #     loss = (kmap.detach() * (outputs - targets).abs()).mean()
    #     loss.backward()
    #     optimizer.step()

    with torch.no_grad():
        outputs = reconstructor(inputs2) #(b,3,h,w)


###############

    if finetune:
        visualization(outputs, 'show/test/raw',iteration=i)
        # outputs = fine_tune(torch.cat([inputs, outputs], dim=1))
        outputs = fine_tune(torch.cat([inputs, outputs], dim=1))


    # evaluation metrics
    psnrout = calc_psnr(outputs, targets)
    # psnrout = calc_psnr(torch.clip(outputs * (1 - kmap), 0, 1), targets * (1 - kmap))
    # ssimout = ssim((outputs * 65535).floor(), (targets * 65535).floor(), data_range=65535, size_average=True)
    ssimout = ssim((outputs * 255).floor(), (targets * 255).floor(), data_range=255, size_average=True)
    lpips = measure.lpips(outputs,targets)

    avg_psnr += psnrout.item()
    avg_ssim += ssimout.item()
    avg_lpips +=lpips

    visualization(kmap * targets,'show/test/kmap3c',iteration=i)
    visualization(kmap ,'show/test/kmap',iteration=i)
    # visualization2('show/test/out3c',outputs3c, iteration=i)
    visualization2('show/test/out',inputs,targets,outputs, iteration=i)





avg_psnr /= len(image_datasets[args.test_filename])
avg_ssim /= len(image_datasets[args.test_filename])
avg_lpips /= len(image_datasets[args.test_filename])
print('PSNR: {:4f}'.format(avg_psnr))
print('SSIM: {:4f}'.format(avg_ssim))
print('lpips: {:4f}'.format(avg_lpips))
