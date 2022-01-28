""" This file contains the definition of the FC-DenseNet103 model.
The architecture and the definitions for all buildings blocks are introduced in
[Jegou et al., 2017](https://arxiv.org/abs/1611.09326).

"""

import torch
import torch.nn as nn
import torch.nn.functional


class FCDenseNet103(nn.Module):
    """ The FC-DenseNet103 semantic segmentation model as defined in Table 2 of (Jegou et al., 2017). """

    def __init__(self, in_channels, n_classes):
        """ Initialize a new FC-DenseNet103.

        Args:
            in_channels (int): Number of channels in the input image.
            n_classes (int): Number of semantic segmentation classes.

        """

        super(FCDenseNet103, self).__init__()

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

        self.tu0 = TransitionUp(in_channels=240)
        self.db5 = DenseBlock(in_channels=896, n_layers=12, growth_rate=16)
        self.tu1 = TransitionUp(in_channels=192)
        self.db6 = DenseBlock(in_channels=656, n_layers=10, growth_rate=16)
        self.tu2 = TransitionUp(in_channels=160)
        self.db7 = DenseBlock(in_channels=464, n_layers=7, growth_rate=16)
        self.tu3 = TransitionUp(in_channels=112)
        self.db8 = DenseBlock(in_channels=304, n_layers=5, growth_rate=16)
        self.tu4 = TransitionUp(in_channels=80)
        self.db9 = DenseBlock(in_channels=192, n_layers=4, growth_rate=16)

        self.conv1 = nn.Conv2d(
            in_channels=64,
            out_channels=n_classes,
            kernel_size=(1, 1),
            bias=False,
        )

        # housekeeping: store the shape of the input to produce output of the exact same size
        self._input_shape = tuple()

    def forward(self, x):
        """ As defined in Figure 1 and Table 2 of (Jegou et al., 2017). """

        x = self._resize_inputs(x)

        out_a = self.conv0(x)

        # beginning the downsampling path
        out_b = self.db0(out_a)
        cat0 = torch.cat([out_a, out_b], dim=1)
        out_a = self.td0(cat0)

        out_b = self.db1(out_a)
        cat1 = torch.cat([out_a, out_b], dim=1)
        out_a = self.td1(cat1)

        out_b = self.db2(out_a)
        cat2 = torch.cat([out_a, out_b], dim=1)
        out_a = self.td2(cat2)

        out_b = self.db3(out_a)
        cat3 = torch.cat([out_a, out_b], dim=1)
        out_a = self.td3(cat3)

        out_b = self.db4(out_a)
        cat4 = torch.cat([out_a, out_b], dim=1)
        out_a = self.td4(cat4)
        # end of the downsampling path

        out_a = self.bottleneck(out_a)

        # beginning of the upsampling path
        out_a = self.tu0(out_a)
        out_a = torch.cat([out_a, cat4], dim=1)
        out_a = self.db5(out_a)

        out_a = self.tu1(out_a)
        out_a = torch.cat([out_a, cat3], dim=1)
        out_a = self.db6(out_a)

        out_a = self.tu2(out_a)
        out_a = torch.cat([out_a, cat2], dim=1)
        out_a = self.db7(out_a)

        out_a = self.tu3(out_a)
        out_a = torch.cat([out_a, cat1], dim=1)
        out_a = self.db8(out_a)

        out_a = self.tu4(out_a)
        out_a = torch.cat([out_a, cat0], dim=1)
        out_a = self.db9(out_a)
        # end of the upsampling path

        out_a = self.conv1(out_a)

        out_a = self._resize_outputs(out_a)

        return out_a

    def _resize_inputs(self, inputs):
        """ Resize the input tensor if its spatial dimensions are not divisible by 2 ** num_transition_down. """

        self._input_shape = inputs.shape[-2:]
        height, width = self._input_shape
        size_factor = 2 ** 5

        if height % size_factor == 0 and width % size_factor == 0:
            return inputs

        new_height, new_width = height, width

        if width % size_factor != 0:
            ratio = width / size_factor
            if ratio - int(ratio) > 0.5:
                new_width = (int(ratio) + 1) * size_factor
            else:
                new_width = int(ratio) * size_factor

        if height % size_factor != 0:
            ratio = height / size_factor
            if ratio - int(ratio) > 0.5:
                new_height = (int(ratio) + 1) * size_factor
            else:
                new_height = int(ratio) * size_factor

        inputs = nn.functional.interpolate(inputs, size=(new_height, new_width), mode='bilinear', align_corners=False)

        return inputs

    def _resize_outputs(self, outputs):
        """ Resize the outputs back to the original size. """

        if outputs.shape[-2:] == self._input_shape:
            return outputs

        height, width = self._input_shape
        outputs = nn.functional.interpolate(outputs, size=(height, width), mode='bilinear', align_corners=False)
        return outputs


class DenseBlock(nn.Module):
    """ As defined in Figure 2 of (Jegou et al., 2017). """

    def __init__(self, in_channels, n_layers, growth_rate):
        super(DenseBlock, self).__init__()

        self.layers = nn.ModuleList()

        in_channels = in_channels
        for i in range(n_layers):
            layer = Layer(in_channels, growth_rate)
            self.layers.append(layer)
            in_channels += growth_rate

    def forward(self, x):
        feature_maps = [x]

        for layer in self.layers:
            in_ = torch.cat(feature_maps, dim=1)
            out = layer(in_)
            feature_maps.append(out)

        # input is not concatenated with the output within the DenseBlock itself
        # (for the upsampling path, to combat feature map explosion)
        # instead, the concatenation is handled in the forward pass of the FC-DenseNet

        out = torch.cat(feature_maps[1:], dim=1)

        return out


class TransitionDown(nn.Module):
    """ As defined in Table 1 of (Jegou et al., 2017). """

    def __init__(self, in_channels):
        super(TransitionDown, self).__init__()

        self.batch_norm = nn.BatchNorm2d(num_features=in_channels)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1),
            bias=False,
        )
        self.dropout = nn.Dropout2d(p=0.2)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.batch_norm(x)
        x = x.relu()
        x = self.conv(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x


class TransitionUp(nn.Module):
    """ As defined in Table 1 of (Jegou et al., 2017). """

    def __init__(self, in_channels):
        super(TransitionUp, self).__init__()

        self.transposed_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 1),
            bias=False,
        )

    def forward(self, x):
        x = self.transposed_conv(x)
        return x


class Layer(nn.Module):
    """ As defined in Table 1 of (Jegou et al., 2017). """

    def __init__(self, in_channels, growth_rate):
        super(Layer, self).__init__()

        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=growth_rate,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False,
        )
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.batch_norm(x)
        x = x.relu()
        x = self.conv(x)
        x = self.dropout(x)

        return x
