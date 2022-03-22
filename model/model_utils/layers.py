"""common deep learning layers for building unet"""
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import utils

# TODO: Complete resnetblock
# TODO: add layer name



class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple):
    """

    def __init__(self, in_channels, out_channels, conv_type, kernel_size=3, order='bcr', num_groups=8, padding=1):
        super(SingleConv, self).__init__()
        if conv_type == '2d':
            conv_ops = create_conv2d
        elif conv_type == '3d':
            conv_ops = create_conv3d
        else:
            raise ValueError('Unknown convolution type')
        for name, module in conv_ops(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, conv_type, encoder, kernel_size=3, order='bcr', num_groups=8, padding=1):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, conv_type, kernel_size, order, num_groups,
                                   padding=padding))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, conv_type, kernel_size, order, num_groups,
                                   padding=padding))


class Decoder(nn.Module):
    # def __init__(self, input_channels, num_class, pool_kernel_size=2, stages=5, root_channel=32, bilinear=True):
    def __init__(self, in_channels, out_channels, conv_type, conv_kernel_size=3, scale_factor=(2, 2, 2), basic_module=DoubleConv,
                 conv_layer_order='gcr', num_groups=8, mode='nearest', padding=1, upsample=True):
        super(Decoder, self).__init__()

        if upsample:
            if basic_module == DoubleConv:
                # if DoubleConv is the basic_module use interpolation for upsampling and concatenation joining
                self.upsampling = InterpolateUpsampling(mode=mode)
                # concat joining
                self.joining = partial(self._joining, join_method='concat')
            else:
                # if basic_module=ExtResNetBlock use transposed convolution upsampling and summation joining
                self.upsampling = TransposeConvUpsampling(in_channels=in_channels, out_channels=out_channels,
                                                          kernel_size=conv_kernel_size, scale_factor=scale_factor)
                # sum joining
                self.joining = partial(self._joining, join_method='concat')
                # adapt the number of in_channels for the ExtResNetBlock
                in_channels = out_channels
        else:
            # no upsampling
            self.upsampling = NoUpsampling()
            # concat joining
            self.joining = partial(self._joining, join_method='concat')
        
        self.basic_module = basic_module(in_channels, out_channels, conv_type,
                                         encoder=False,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, join_method):
        if join_method == 'concat':
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x 
        

class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode='nearest'):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class TransposeConvUpsampling(AbstractUpsampling):
    """
    Args:
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True
    """

    def __init__(self, in_channels=None, out_channels=None, kernel_size=3, scale_factor=(2, 2, 2)):
        # make sure that the output size reverses the MaxPool3d from the corresponding encoder
        upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor,
                                      padding=1)
        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        return x

class _DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, stride=1, batch_norm=True, activation='relu'):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        self.conv1 = Conv_Bn_Activation(in_channels=in_channels, 
                                        out_channels=mid_channels, 
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        batch_norm=batch_norm,
                                        activation=activation)
        self.conv2 = Conv_Bn_Activation(in_channels=mid_channels, 
                                        out_channels=out_channels, 
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        batch_norm=batch_norm,
                                        activation=activation)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, batch_norm=True, activation='relu'):
        super().__init__()
        padding = (kernel_size - 1) // 2
        modules = []
        modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
        if batch_norm is True:
            modules.append(nn.BatchNorm2d(out_channels))
        if activation == 'relu':
            modules.append(nn.ReLU(inplace=True))
        self.conv_bn_activation = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv_bn_activation(x)


def conv3d(in_channels, out_channels, kernel_size, bias, padding):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

    
def create_conv3d(in_channels, out_channels, kernel_size, order, num_groups, padding):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")
    return modules


def create_conv2d(in_channels, out_channels, kernel_size, order, num_groups, padding):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm2d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm2d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")
    return modules


class MultiLayerPerceptron(nn.Module):
    def __init__(self, structure, activation=None, out_activation=None, *args, **kwargs):
        super(MultiLayerPerceptron, self).__init__()
        self.mlp = torch.nn.Sequential()
        self.activation = activation
        self.out_activation = out_activation
        assert isinstance(structure, (list, tuple)), 'Model structure "structure" should be list or tuple'
        assert len(structure) > 1, 'The length of structure should be at least 2 to define linear layer'

        for idx in range(len(structure)-1):
            in_channels, out_channels = structure[idx], structure[idx+1]
            self.mlp.add_module(f"fc{idx+1}", torch.nn.Linear(in_channels, out_channels))
            if self.activation is not None and idx+1 < len(structure)-1:
                self.mlp.add_module(f"{self.activation}{idx+1}", get_activation(self.activation))

        if self.out_activation is not None:
            out_dix = idx + 2 if self.out_activation == self.activation else 1
            self.mlp.add_module(f"{self.out_activation}{out_dix}", get_activation(self.out_activation))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x

        
# class Encoder(nn.Module):
#     def __init__(self, in_channels, out_channels, basic_module=ConventionalConv, conv_kernel_size=3, padding=1, pooling='max', pool_kernel_size=2):
#         super().__init__()
#         assert pooling in ['max', 'avg']
#         if pooling == 'max':
#             self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
#         else:
#             self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)

#         self.basic_module = basic_module(in_channels, out_channels,
#                                          encoder=True,
#                                          kernel_size=conv_kernel_size,
#                                          order=conv_layer_order,
#                                          num_groups=num_groups,
#                                          padding=padding)
#     def forward(self, x):
#         if self.pooling is not None:
#             x = self.pooling(x)
#         x = self.basic_module(x)
#         return x


# if __name__ == "__main__":
#     # print(Conv_Bn_Activation(32, 64))
#     print(DoubleConv(32, 64))