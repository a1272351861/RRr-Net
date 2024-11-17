"""

finetune前面加一个自适应各种图片尺寸的transformer
"""

import os
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from modules2.model import UNet
# from modules2.RGBformer.RGBformer import RGBformer
from modules2.restomer.RGBformerCNN import RGBformerCNN,RGBformer
from modules2.isp_model_main import UNet_isp, UNet_reverse
# from dataset_raw import DatasetRAW,DatasetRAWTest
from utils.dataset_lol import DatasetRAW_allin,calculate_params,DatasetRAW_exp, visualization,Dataset_sketch,visualization2,DatasetRAWTest
from utils.util import poolfeat, upfeat, calc_psnr, Measure,get_kmap_from_prob,my_collate
from pytorch_msssim import ssim
from tensorboardX import SummaryWriter

class Coach_sketch:
    def __init__(self,num,device,mode='train',batchsize=None):
        self.checkpoint = False
        self.freeze_sampler = False
        self.visualization_train = True
        self.superpixel_reverser_loss = False
        # notarget = True
        # nokmap = '3' #1:no kmap 2:no kmap,kmap*out
        self.isp = False
        self.bayer = True
        self.metadata = False

        self.task = 'lol'##'lol''exp''raw''sketch'
        self.num = str(num)
        self.device = device
        self.batchsize = batchsize
        self.mode = mode
        self.test_dir = 'validation' #eval_v2,Test,validation

        parser = argparse.ArgumentParser(description='Learn samples: full float')
        '''
        _reverser-raw-pretrain-nofreeze
        
        '''
        if self.task == 'lol':
            parser.add_argument(
                '--data-dir', default='/home/zdx/my_shared_data_folder/LOLdataset', type=str, help='folder of training and validation images')
                # '--data-dir', default='/home/zdx/my_shared_data_folder/LOL-v2/Real_captured', type=str, help='folder of training and validation images')
                # '--data-dir', default='/home/zdx/my_shared_data_folder/LOL-v2/Synthetic', type=str, help='folder of training and validation images')
            # head = 'lolv1-224patch-baseline-noaugmentation_RGBformer-CNN-rgb-ft24-num4_r&finetune+isp_faster3_nofreeze_originalsampler_finetuner-ft8-skip_reconstruct-7in-8ft'
            head = 'lolv1-224patch-Bayer-noaugmentation_RGBformer-CNN-rgb-ft24-num4_r&finetune+isp_faster3_nofreeze_originalsampler_finetuner-ft8-skip_reconstruct-7in-8ft'
            # head = 'lolv1train-VEtest-224patch-baseline-noaugmentation_RGBformer-CNN-rgb-ft24-num4_r&finetune+isp_faster3_nofreeze_originalsampler_finetuner-ft8-skip_reconstruct-7in-8ft'
        elif self.task == 'exp':
            parser.add_argument(
                '--data-dir', default='/home/zdx/my_shared_data_folder/Exposure_Correction', type=str, help='folder of training and validation images')
            head = 'exp_raw_rs+isp_fasterblocks3_loss123'
        elif self.task == 'raw':
            parser.add_argument(
                '--data-dir', default='/home/zdx/paper_experiments/content-aware-metadata-main/SamsungNX2000', type=str, help='folder of training and validation images')
            head = 'samsung_raw_rs+isp_fasterblocks3_Superpixel+reverser-loss_pretrain'
        elif self.task == 'sketch':
            parser.add_argument(
                '--data-dir', default='/home/zdx/my_shared_data_folder/LOLdataset', type=str, help='folder of training and validation images')
            head = 'lol_r&finetune+isp_fasterblocks3_nofreeze_sketch_finetuner-ft4-skip'

        parser.add_argument('--checkpoint-dir', default=\
        '/home/zdx/paper_experiments/content-aware-metadata-main/models/lolv1-224patch-noaugmentation_RGBformer-CNN-rgb-ft24-num4_r&finetune+isp_faster3_nofreeze_originalsampler_finetuner-ft8-skip_reconstruct-7in-8ft_lr0.001_e120_b16_ft8_', type=str, help='folder of checkpoint')

        parser.add_argument('--test-checkpoint-dir', default= \
'/home/zdx/paper_experiments/content-aware-metadata-main/models/Sony-512patch-noaugmentation_RGBformer-CNN-rgb-ft24-num4_r&finetune+isp_faster3_nofreeze_originalsampler_finetuner-ft8-skip_reconstruct-7in-8ft_lr0.001_e1200_b4_ft8_'
                            ,type=str, help='folder of checkpoint')

        parser.add_argument(
            '--file-type', default='jpg', type=str, help='image file type (png or tif)')
        parser.add_argument(
            '--num-epochs', type=int, default=120, help='number of epochs')
        parser.add_argument(
            '--tboard-freq', type=int, default=200, help='frequency of writing to tensorboard')
        parser.add_argument(
            '--k', type=float, default=1.5625, help='percentage of samples to pick')
        parser.add_argument(
            '--patch-size', type=int, default=224, help='patch size')
        parser.add_argument(
            '--init-features', type=int, default=8, help='init_features of UNet')
        parser.add_argument(
            '--stride', type=int, default=52, help='stride when cropping patches')
        parser.add_argument(
            '--batch-size', type=int, default=10, help='batch size')
        parser.add_argument(
            '--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument(
            '--lambda-slic', type=float, default=0.0001, help='weight of SLIC loss')
        parser.add_argument(
            '--slic-alpha', type=float, default=0.2, help='alpha in SLIC loss')
        parser.add_argument(
            '--slic-m', type=float, default=10.0, help='weight m in SLIC loss')
        parser.add_argument(
            '--lambda-meta', type=float, default=0.01, help='weight of meta loss')
        parser.add_argument(
            '--inner-lr', type=float, default=0.001, help='learning rate of inner loop')
        parser.add_argument(
            '--inner-steps', type=int, default=5, help='number of update steps in inner loop')
        parser.add_argument(
            '--extra', type=str, default='', help='extra identifier for save folder name')

        # parser.add_argument('--gpu_id', type=str, default=1)
        self.args = parser.parse_args()
        # torch.backends.cudnn.enabled = False
        # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        # torch.cuda.set_device(self.args.gpu_id)
        self.device = torch.device("cuda:{}".format(self.device) if torch.cuda.is_available() else "cpu")
        if self.batchsize:
            self.args.batch_size =self.batchsize

        grid_size = self.args.patch_size ** 2 * self.args.k / 100
        if np.sqrt(grid_size) != int(np.sqrt(grid_size)):
            print('Warning: superpixel grid seeds may not match the percentage of samples.')
        self.grid_size = self.args.patch_size // int(np.sqrt(grid_size))

        savefoldername = (
            head
            + '_lr' + str(self.args.lr)
            + '_e' + str(self.args.num_epochs)
            + '_b' + str(self.args.batch_size)
            + '_ft' + str(self.args.init_features)
            + '_' + self.args.extra
        )

        self.writer = SummaryWriter(os.path.join('./logs', savefoldername))
        self.mysavepath = os.path.join('./models', savefoldername)

        # if not(os.path.exists(self.mysavepath) and os.path.isdir(self.mysavepath)):
        #     os.makedirs(self.mysavepath)

        if self.task == 'lol' or self.task == 'raw':
            if self.mode =='train':
                self.image_datasets = {x: DatasetRAW_allin(os.path.join(self.args.data_dir, x), self.args.batch_size, self.args.patch_size, self.args.stride, to_gpu=self.device, ftype=self.task)
                                  for x in ['train', 'validation']}
            else:
                self.test_dataset = DatasetRAWTest(os.path.join(self.args.data_dir, self.test_dir))
        elif self.task == 'exp':
            if self.mode == 'train':
                self.image_datasets = {x: DatasetRAW_exp(os.path.join(self.args.data_dir, x), self.args.batch_size, self.args.patch_size,self.args.stride, to_gpu=self.device, ftype=self.args.file_type)
                                  for x in ['train', 'validation']}
            else:
                self.test_dataset = DatasetRAWTest(os.path.join(self.args.data_dir, self.test_dir))
        elif self.task == 'sketch':
            if self.mode == 'train':
                self.image_datasets = {x: Dataset_sketch(os.path.join(self.args.data_dir, x), self.args.batch_size, self.args.patch_size,self.args.stride, train=True, ftype=self.args.file_type)
                                  for x in ['train', 'validation']}
            else:
                self.test_dataset = Dataset_sketch(os.path.join(self.args.data_dir, self.test_dir), self.args.batch_size,
                                                   self.args.patch_size, self.args.stride, train=False,
                                                   ftype=self.args.file_type)

        #train dataloader
        # self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=self.args.batch_size,
        #                                               shuffle=True, num_workers=0,collate_fn=my_collate)
        #               for x in ['train', 'validation']}
        if self.mode == 'train':
            self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=self.args.batch_size,
                                                      shuffle=True, num_workers=0)
                      for x in ['train', 'validation']}
        ###test dataloader
        else:
            self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                          batch_size=1,shuffle=False, num_workers=0)
        # if self.sampler:

        self.sampler = UNet(in_channels=6, out_channels=9, init_features=self.args.init_features, sigmoid=False).to(self.device)
        # self.sampler = None

            # self.sampler = UNet_reverse(in_channels=3, out_channels=4, init_features=self.args.init_features, sigmoid=False) ##2 return

        if self.metadata:
            self.reconstructor = UNet_isp(in_channels=7, out_channels=3, init_features=self.args.init_features).to(self.device)
            print('with Metadata')
        else:
            self.reconstructor = UNet(in_channels=3, out_channels=3, init_features=self.args.init_features).to(self.device)

        if self.bayer:
            self.bayer = RGBformer(dim=24, num_blocks=4, ).to(self.device)
            print('with RGBformer')

        if self.isp:
            self.fine_tune = UNet_isp(in_channels=6, out_channels=3, init_features=8).to(self.device)
            print('with ISP')
        else:
            self.fine_tune = UNet(in_channels=6, out_channels=3, init_features=8).to(self.device)

        if self.checkpoint :
            # self.reconstructor.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'best_reconstructor.pt')))
            self.sampler.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'best_sampler.pt')))
            self.reconstructor.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'best_reconstructor.pt')))
            if self.bayer:
                self.bayer.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'best_bayer.pt')))

            self.fine_tune.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'best_finetune.pt')))
            if self.freeze_sampler:
                for param in self.sampler.parameters():
                    param.requires_grad = False
                print('self.sampler checkpoint freeze')
                # for param in self.reconstructor.parameters():
                #     param.requires_grad = False
                # print('self.reconstructor checkpoint freeze')
            print('checkpoint load finished')

        params = list(self.sampler.parameters())
        params += list(self.reconstructor.parameters())
        if self.bayer:
            params += list(self.bayer.parameters())
        params += list(self.fine_tune.parameters())

        self.optimizer = optim.Adam(filter(lambda p : p.requires_grad, params), lr=self.args.lr)
        # print(self.bayer)
        # print(self.sampler, self.reconstructor, self.fine_tune)
        # print(params)
        calculate_params(params)

        self.measure = Measure()

    def train(self):

        dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'validation']}

        epoch_loss = {x: 0.0 for x in ['train', 'validation']}
        epoch_recon = {x: 0.0 for x in ['train', 'validation']}
        epoch_slic = {x: 0.0 for x in ['train', 'validation']}
        epoch_psnr = {x: 0.0 for x in ['train', 'validation']}
        epoch_lpips = {x: 0.0 for x in ['train', 'validation']}
        # training loop starts here
        since = time.time()
        best_loss = 10 ** 6
        best_psnr = 0.0


        for epoch in range(self.args.num_epochs):
            running_loss_tboard = 0.0
            running_recon_tboard = 0.0
            running_slic_tboard = 0.0
            running_psnr_tboard = 0.0
            running_lpips_tboard = 0.0


            print('Epoch {}/{}'.format(epoch, self.args.num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.sampler.train()
                    self.reconstructor.train()
                    if self.bayer:
                        self.bayer.train()
                    self.fine_tune.train()
                else:
                    self.sampler.eval()
                    self.reconstructor.eval()
                    if self.bayer:
                        self.bayer.eval()

                    self.fine_tune.eval()

                running_loss = 0.0
                running_recon = 0.0
                running_slic = 0.0
                running_psnr = 0.0
                running_lpips = 0.0

                # counter for tboard
                if phase == 'train':
                    i = 0

                # Iterate over data.
                for inputs, targets  in self.dataloaders[phase]:
                    ##input (b,3,patch,patch)
                    # inputs, targets = inputs.cuda(), targets.cuda()
                    inputs, targets= inputs.to(self.device), targets.to(self.device)
                    # inputs, targets,sketch = inputs.to(self.device), targets.to(self.device),sketch.to(self.device)

                    if phase == 'train':
                        i += 1

                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):

                        output,slic = self.forward(inputs,targets,i,mode=phase)

                        recon = F.l1_loss(output, targets)

                        # if self.superpixel_reverser_loss:
                            # recon_reverser = F.l1_loss(outputs3c, targets)
                        #     recon = 0.5*recon + 0.5*recon_reverser

                        # visualization2('show/trian/',targets,outputs3c,outputs2)
                        if self.visualization_train:
                            # visualization(inputs,'show/train%s/inputs' % self.num, iteration=i)
                            # visualization(targets,'show/train%s/targets' % self.num, iteration=i)
                            visualization('show/train%s/output' % self.num,output, iteration=i)

                            # visualization(sketch, 'show/train1/sketch',i)
                            # if finetune:

                        # psnrout = calc_psnr(torch.clip(outputs2 * (1 - kmap), 0, 1), targets * (1 - kmap))
                        psnrout = calc_psnr(output,targets)
                        lpips_o = self.measure.lpips(output,targets)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss = recon + self.args.lambda_slic * slic

                            loss.backward()
                            self.optimizer.step()

                            running_loss_tboard += loss.item()
                            running_recon_tboard += recon.item()
                            running_psnr_tboard += psnrout.item()
                            running_lpips_tboard += lpips_o

                            if i % self.args.tboard_freq == self.args.tboard_freq - 1:

                                # ...log the running loss
                                self.writer.add_scalar('loss',
                                                running_loss_tboard / self.args.tboard_freq,
                                                epoch * len(self.dataloaders[phase]) + i)

                                self.writer.add_scalar('recon',
                                                running_recon_tboard / self.args.tboard_freq,
                                                epoch * len(self.dataloaders[phase]) + i)

                                self.writer.add_scalar('psnr',
                                                running_psnr_tboard / self.args.tboard_freq,
                                                epoch * len(self.dataloaders[phase]) + i)
                                self.writer.add_scalar('lpips',
                                                running_lpips_tboard / self.args.tboard_freq,
                                                epoch * len(self.dataloaders[phase]) + i)

                                running_loss_tboard = 0.0
                                running_recon_tboard = 0.0
                                running_psnr_tboard = 0.0
                                running_lpips_tboard = 0.0
                        else:
                            # loss = recon
                            loss = recon + self.args.lambda_slic * slic

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_recon += recon.item() * inputs.size(0)
                    running_slic += slic.item() * inputs.size(0)
                    running_psnr += psnrout.item() * inputs.size(0)
                    running_lpips += lpips_o * inputs.size(0)

                epoch_loss[phase] = running_loss / dataset_sizes[phase]
                epoch_recon[phase] = running_recon / dataset_sizes[phase]
                epoch_slic[phase] = running_slic / dataset_sizes[phase]
                # epoch_recon_meta[phase] = running_recon_meta / dataset_sizes[phase]
                epoch_psnr[phase] = running_psnr / dataset_sizes[phase]
                epoch_lpips[phase] = running_lpips / dataset_sizes[phase]


            if phase == 'validation':
                # ...log the running loss
                self.writer.add_scalars('loss',
                                  {'train': epoch_loss['train'],'validation': epoch_loss['validation']},
                                  (epoch+1) * len(self.dataloaders['train']))

                self.writer.add_scalars('recon',
                                  {'train': epoch_recon['train'],'validation': epoch_recon['validation']},
                                  (epoch+1) * len(self.dataloaders['train']))

                self.writer.add_scalars('psnr',
                                  {'train': epoch_psnr['train'],'validation': epoch_psnr['validation']},
                                  (epoch+1) * len(self.dataloaders['train']))

                self.writer.add_scalars('lpips',
                                  {'train': epoch_lpips['train'],'validation': epoch_lpips['validation']},
                                  (epoch+1) * len(self.dataloaders['train']))


            print('{} Loss: {:.6f}, Recon: {:.6f}, SLIC: {:.6f}, PSNR: {:.4f}'.format(
                # phase, epoch_loss[phase], epoch_recon[phase], epoch_slic[phase], epoch_recon_meta[phase], epoch_psnr[phase]))
                phase, epoch_loss[phase], epoch_recon[phase], epoch_slic[phase], epoch_psnr[phase],epoch_lpips[phase]))

            # deep copy the sampler
            if phase == 'validation' and epoch_loss[phase] < best_loss:
                best_loss = epoch_loss[phase]
                best_psnr = epoch_psnr[phase]
                best_lpips = epoch_lpips[phase]
                print('best_psnr',best_psnr)
                print('best_lpips',best_lpips)
                if not (os.path.exists(self.mysavepath) and os.path.isdir(self.mysavepath)):
                    os.makedirs(self.mysavepath)
                    print("makedir at-----------", self.mysavepath)
                torch.save(self.sampler.state_dict(), os.path.join(self.mysavepath, 'best_sampler.pt'))
                torch.save(self.reconstructor.state_dict(), os.path.join(self.mysavepath, 'best_reconstructor.pt'))
                if self.bayer:
                    torch.save(self.bayer.state_dict(), os.path.join(self.mysavepath, 'best_bayer.pt'))
                torch.save(self.fine_tune.state_dict(), os.path.join(self.mysavepath, 'best_finetune.pt'))
                with open(os.path.join(self.mysavepath, 'timestamp.txt'), 'a') as f:
                    f.write(
                        '**Best**: Step - {}, Loss - {:.3f} \nbest_psnr:{}\nbest_lpips:{}\n\n'.format(epoch,best_loss,
                                                                                                        best_psnr,best_lpips))


        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:4f}'.format(best_loss))
        print('Best val psnr: {:4f}'.format(best_psnr))


    def forward(self,inputs,targets,idx,mode):

        if self.metadata:
            slic = 0
        else:
            slic = torch.tensor(0).to(self.device)
        outputs = self.sampler(torch.cat((inputs, targets), 1))
        prob = F.softmax(outputs, 1)  # b,9,p,p
        #####test 用不上
        if mode == 'train' or mode =='validation' and self.metadata:
            coords = torch.stack(torch.meshgrid(torch.arange(self.args.patch_size, device=self.device),
                                                torch.arange(self.args.patch_size, device=self.device)), 0)
            coords = coords[None].repeat(inputs.shape[0], 1, 1, 1).float()  # b,2,128,128
            inputs_targets_rgbxy = torch.cat([255 * inputs, 255 * targets, coords], 1)
            pooled_labxy = poolfeat(inputs_targets_rgbxy, prob, self.grid_size, self.grid_size, self.device)  # b,8,16,16
            reconstr_feat = upfeat(pooled_labxy, prob, self.grid_size, self.grid_size)  # b,8,p,p

            slic = self.args.slic_alpha * F.mse_loss(reconstr_feat[:, :3, :, :], inputs_targets_rgbxy[:, :3, :, :]) + \
                   (1 - self.args.slic_alpha) * F.mse_loss(reconstr_feat[:, 3:6, :, :],
                                                           inputs_targets_rgbxy[:, 3:6, :, :]) + \
                   self.args.slic_m ** 2 / self.grid_size ** 2 * F.mse_loss(reconstr_feat[:, 6:, :, :],
                                                                            inputs_targets_rgbxy[:, 6:, :, :])

        ####################
        if self.metadata:
            # sampling process
            kmap = get_kmap_from_prob(prob, self.grid_size).to(self.device)  # b,1,p,p
            inputs2 = torch.cat([ inputs,  kmap *targets, kmap], dim=1)  ####
        else:
            inputs2 = inputs

        outputs = self.reconstructor(inputs2)
        if self.bayer:
            outputs = self.bayer(outputs)


        # outputs2 = self.fine_tune(outputs2)
        # visualization('show/train%s/raw'%self.num,outputs, idx)
        # visualization(sketch, 'show/train%s/sketch'%self.num,idx)
        # visualization(sketch*target, 'show/train%s/sketchxtarget'%self.num,idx)
        # visualization('show/test%s/kmapxtarget'%self.num,kmap*targets, idx)
        # visualization('show/test%s/kmap'%self.num,kmap, idx)
        outputs = self.fine_tune(torch.cat([inputs, outputs], dim=1))
        return outputs,slic

    def test_image_generation(self):
        print('-----------------testing----------------')
        self.sampler.load_state_dict(torch.load(os.path.join(self.args.test_checkpoint_dir, 'best_sampler.pt')))
        self.reconstructor.load_state_dict(torch.load(os.path.join(self.args.test_checkpoint_dir, 'best_reconstructor.pt')))
        if self.bayer:
            self.bayer.load_state_dict(torch.load(os.path.join(self.args.test_checkpoint_dir, 'best_bayer.pt')))
            self.bayer.eval()
        self.fine_tune.load_state_dict(torch.load(os.path.join(self.args.test_checkpoint_dir, 'best_finetune.pt')))

        self.sampler.eval()
        self.reconstructor.eval()

        self.fine_tune.eval()

        avg_psnr = 0
        avg_ssim = 0
        avg_lpips = 0

        for i, (inputs, targets) in enumerate(self.test_dataloader):
            inputs, targets= inputs.to(self.device), targets.to(self.device)
            B, C, H, W = inputs.shape
            # inputs, targets ,sketch= inputs.to(self.device), targets.to(self.device),sketch.to(self.device)
            # inputs = F.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)
            # targets = F.interpolate(targets, size=(224, 224), mode='bilinear', align_corners=False)

            outputs,_ = self.forward(inputs,targets,i,mode='test')

            # outputs = F.interpolate(outputs, size=(H, W), mode='bilinear', align_corners=False)
            # inputs = F.interpolate(inputs, size=(H, W), mode='bilinear', align_corners=False)
            # targets = F.interpolate(targets, size=(H, W), mode='bilinear', align_corners=False)

            visualization2('show/test%s/out'%self.num, inputs, targets, outputs, iteration=i)
            visualization('show/test%s/outputs'%self.num, outputs, iteration=i)
            visualization('show/test%s/targets'%self.num, targets, iteration=i)
            visualization('show/test%s/inputs'%self.num, inputs, iteration=i)

            # evaluation metrics
            psnrout = calc_psnr(outputs, targets)
            # ssimout = ssim((outputs * 65535).floor(), (targets * 65535).floor(), data_range=65535, size_average=True)
            ssimout = ssim((outputs * 255).floor(), (targets * 255).floor(), data_range=255, size_average=True)
            lpips = self.measure.lpips(outputs, targets)

            avg_psnr += psnrout.item()
            avg_ssim += ssimout.item()
            avg_lpips += lpips

            # visualization(kmap * targets, 'show/test/kmap3c', iteration=i)
            # visualization(kmap, 'show/test/kmap', iteration=i)
            # visualization2('show/test/out3c',outputs3c, iteration=i)
            # visualization('show/test%s/sketchxtarget'%self.num,sketch*target, iteration=i)


        avg_psnr /= len(self.test_dataset)
        avg_ssim /= len(self.test_dataset)
        avg_lpips /= len(self.test_dataset)
        print('PSNR: {:4f}'.format(avg_psnr))
        print('SSIM: {:4f}'.format(avg_ssim))
        print('lpips: {:4f}'.format(avg_lpips))



if __name__=='__main__':
    coach = Coach_sketch(num=0,device=0,mode='test')###test只能cuda0
    # coach = Coach_sketch(num=0,device=0,mode='train')
    # coach.train()
    coach.test_image_generation()