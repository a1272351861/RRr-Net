import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import math
import sys

# print(sys.path)
# print(len(sys.path))
# import blocks
from timm.models.layers import trunc_normal_
# from model.blocks import CBlock_ln, SwinTransformerBlock
# from IAT_enhance.model.blocks import CBlock_ln, SwinTransformerBlock, DCNv3_block
# from .blocks import CBlock_ln, SwinTransformerBlock
from .blocks import CBlock_ln, SwinTransformerBlock,fasterBlock
# from .global_net import Global_pred


# Short Cut Connection on Final Layer
class Local_pred_S(nn.Module):
    def __init__(self, in_dim=3, dim=16, number=3, type='faster'):
        super(Local_pred_S, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(in_dim, dim, 3, padding=1, groups=1)  # 'same'conv
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # main blocks
        block = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)  # head number
        if type == 'ccc':
            blocks = [CBlock_ln(dim, drop_path=0.01), CBlock_ln(dim, drop_path=0.05), CBlock_ln(dim, drop_path=0.1)]
            # blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        elif type == 'ttt':
            blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        elif type == 'cct':
            blocks1, blocks2 = [block, block, block_t], [block, block, block_t]

        elif type == 'increase blocks from 4 to number':
            # blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
            blocks1, blocks2 = [CBlock_ln(dim, drop_path=0.01) for _ in range(number)], [CBlock_ln(16, drop_path=0.01)
                                                                                        for _ in range(number)]
        # elif type == 'DCNv3':
        #     # blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        #     blocks = [DCNv3_block(dim, dim) for _ in range(number)]
        elif type == 'faster':
            # blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
            blocks = [fasterBlock(dim=dim,n_div=4,)for _ in range(number)]
        #    block1 = [CBlock_ln(16), nn.Conv2d(16,24,3,1,1)]
        self.factor_blocks = nn.Sequential(*blocks)
        # self.mul_blocks = nn.Sequential(*blocks1)
        # self.add_blocks = nn.Sequential(*blocks2)

        # self.mul_end = nn.Sequential(nn.Conv2d(dim, in_dim, 3, 1, 1), nn.ReLU())
        # self.add_end = nn.Sequential(nn.Conv2d(dim, in_dim, 3, 1, 1), nn.Tanh())
        self.factor_end = nn.Sequential(nn.Conv2d(dim, in_dim, 3, 1, 1), nn.Tanh())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, img):
        img = self.relu(self.conv1(img))  # (b,dim16,400,600)
        # short cut connection
        # mul = self.mul_blocks(img) + img
        # add = self.add_blocks(img) + img
        factor = self.factor_blocks(img) + img
        # mul = self.mul_end(mul)
        # add = self.add_end(add)
        factor = self.factor_end(factor) # (b,in_dim=3,400,600)

        # return mul, add
        return factor




# if __name__ == "__main__":
#     os.environ['CUDA_VISIBLE_DEVICES'] = '3'
#     img = torch.Tensor(1, 3, 400, 600)
#     net = IAT()
#     print('total parameters:', sum(param.numel() for param in net.parameters()))
#     _, _, high = net(img)
