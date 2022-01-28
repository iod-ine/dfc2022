""" An encoder part of the FC-DenseNet103. This will be the learner in the self-supervised setting. """

import torch
import torch.nn as nn
import torch.nn.functional

from .fcdensenet103 import DenseBlock, TransitionDown


class FCDenseNet103Encoder(nn.Module):
    """ The encoder part of the  FC-DenseNet103 semantic segmentation model. """

    def __init__(self, in_channels):
        """ Initialize a new FC-DenseNet103Encoder.

        Args:
            in_channels (int): Number of channels in the input image.

        """

        super(FCDenseNet103Encoder, self).__init__()

        self.conv0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=48,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False,
        )

        self.db0 = DenseBlock(in_channels=48, n_layers=4, growth_rate=16)
        self.td0 = TransitionDown(in_channels=112)
        self.db1 = DenseBlock(in_channels=112, n_layers=5, growth_rate=16)
        self.td1 = TransitionDown(in_channels=192)
        self.db2 = DenseBlock(in_channels=192, n_layers=7, growth_rate=16)
        self.td2 = TransitionDown(in_channels=304)
        self.db3 = DenseBlock(in_channels=304, n_layers=10, growth_rate=16)
        self.td3 = TransitionDown(in_channels=464)
        self.db4 = DenseBlock(in_channels=464, n_layers=12, growth_rate=16)
        self.td4 = TransitionDown(in_channels=656)

        self.bottleneck = DenseBlock(in_channels=656, n_layers=15, growth_rate=16)

        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(in_features=960, out_features=480),
            nn.ReLU(),
            nn.Linear(in_features=480, out_features=128),
        )

    def forward(self, x):
        """ As defined in Figure 1 and Table 2 of (Jegou et al., 2017). """

        out_a = self.conv0(x)

        out_b = self.db0(out_a)
        out_a = self.td0(torch.cat([out_a, out_b], dim=1))

        out_b = self.db1(out_a)
        out_a = self.td1(torch.cat([out_a, out_b], dim=1))

        out_b = self.db2(out_a)
        out_a = self.td2(torch.cat([out_a, out_b], dim=1))

        out_b = self.db3(out_a)
        out_a = self.td3(torch.cat([out_a, out_b], dim=1))

        out_b = self.db4(out_a)
        out_a = self.td4(torch.cat([out_a, out_b], dim=1))

        out_a = self.bottleneck(out_a)

        out_a = self.projection_head(out_a)

        return out_a
