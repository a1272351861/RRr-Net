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

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=8, sigmoid=True):
        super(UNet, self).__init__()
        self.sigmoid = sigmoid

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        # x:(b,c,h,w)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.conv(dec1)
        if self.sigmoid:
            return torch.sigmoid(dec1)
        return dec1

    def forward_meta(self, x, vars):
        enc1, vars = self.forward_block(x, vars, self.encoder1)
        enc2, vars = self.forward_block(self.pool1(enc1), vars, self.encoder2)
        enc3, vars = self.forward_block(self.pool2(enc2), vars, self.encoder3)
        enc4, vars = self.forward_block(self.pool3(enc3), vars, self.encoder4)

        bottleneck, vars = self.forward_block(self.pool4(enc4), vars, self.bottleneck)

        dec4, vars = self.forward_convtranspose2d(bottleneck, vars)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4, vars = self.forward_block(dec4, vars, self.decoder4)
        dec3, vars = self.forward_convtranspose2d(dec4, vars)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3, vars = self.forward_block(dec3, vars, self.decoder3)
        dec2, vars = self.forward_convtranspose2d(dec3, vars)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2, vars = self.forward_block(dec2, vars, self.decoder2)
        dec1, vars = self.forward_convtranspose2d(dec2, vars)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1, vars = self.forward_block(dec1, vars, self.decoder1)
        dec1 = self.forward_conv2d_last(dec1, vars)
        if self.sigmoid:
            return torch.sigmoid(dec1)
        return dec1

    def forward_block(self, x, vars, block):
        x, vars = self.forward_conv2d(x, vars)
        x = F.batch_norm(x, block[1].running_mean, block[1].running_var, vars[0], vars[1], training=True)
        x = F.relu(x, inplace=True)
        x, vars = self.forward_conv2d(x, vars[2:])
        x = F.batch_norm(x, block[4].running_mean, block[4].running_var, vars[0], vars[1], training=True)
        x = F.relu(x, inplace=True)
        return x, vars[2:]

    def forward_conv2d(self, x, vars):
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        x = F.conv2d(x, vars[0])
        return x, vars[1:]

    def forward_conv2d_last(self, x, vars):
        x = F.conv2d(x, vars[0], vars[1])
        return x

    def forward_convtranspose2d(self, x, vars):
        x = F.conv_transpose2d(x, vars[0], vars[1], stride=2)
        return x, vars[2:]


    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                            padding_mode='replicate'
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                            padding_mode='replicate'
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )