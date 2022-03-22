import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import utils
from model.model_utils import layers
from model.model_utils import backbones
number_of_features_per_level = utils.number_of_features_per_level
SingleConv = layers.SingleConv
DoubleConv = layers.DoubleConv
_DoubleConv = layers._DoubleConv
Decoder = layers.Decoder
Conv_Bn_Activation = layers.Conv_Bn_Activation
creat_torchvision_backbone = backbones.creat_torchvision_backbone

class UNet_2d_backbone(nn.Module):
    def __init__(self, in_channels, out_channels, basic_module, f_maps=64, layer_order='bcr',
                 num_groups=8, num_levels=4, is_segmentation=True, testing=False,
                 conv_kernel_size=3, pool_kernel_size=2, conv_padding=1, backbone='resnet50', pretrained=True, **kwargs):
        super(UNet_2d_backbone, self).__init__()
        self.out_channels = out_channels

        if in_channels != 3 and pretrained:
            logging.info('Reinitialized first layer')

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.encoders = create_encoder(in_channels, backbone, pretrained=pretrained, final_flatten=False)
        
        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, 
                                        num_groups, upsample=True, conv_type='2d')

        
        self.conv = basic_module(f_maps[0], f_maps[0]//2, conv_type='2d',
                                encoder=False,
                                kernel_size=conv_kernel_size,
                                order='bcr',
                                num_groups=num_groups,
                                padding=conv_padding)
        # self.classifier = Conv_Bn_Activation(in_channels=f_maps[0], out_channels=self.out_channels, activation=None)
        self.classifier = SingleConv(f_maps[0]//2, out_channels, conv_type='2d', kernel_size=1, order='bc')

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        # # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        # # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        # if self.testing and self.final_activation is not None:
        #     x = self.final_activation(x)
        # return x
        
        x = F.interpolate(x, size=x.size()[2]*2, mode='bilinear')
        x = self.conv(x)
        x = self.classifier(x)
        return nn.Sigmoid()(x)



    # def forward(self, x):
    #     f1 = self.conv1(x)
    #     f2 = self.conv2(f1)
    #     f3 = self.conv3(f2)
    #     f4 = self.conv4(f3)
    #     f5 = self.conv5(f4)

    #     features = [f1, f2, f3, f4, f5]
    #     x = self.decoder(features)
    #     pass
    #     return x


class UNet_2d(nn.Module):
    def __init__(self, input_channels, num_class, pool_kernel_size=2, stages=5, root_channel=32, bilinear=True):
        super(UNet_2d, self).__init__()
        self.bilinear = bilinear
        self.name = '2d_unet'

        # model2 = nn.Sequential(collections.OrderedDict([
        #         ('conv1', Conv_Bn_Activation(in_channels=root_channel, out_channels=root_channel)),
        #         ('conv2', nn.ReLU()),
        #         ('conv3', nn.Conv2d(20,64,5)),
        #         ('conv4', nn.ReLU())
        #         ]))

        self.conv1 = _DoubleConv(in_channels=input_channels, out_channels=root_channel, mid_channels=root_channel)
        self.conv2 = _DoubleConv(in_channels=root_channel, out_channels=root_channel*2)
        self.conv3 = _DoubleConv(in_channels=root_channel*2, out_channels=root_channel*4)
        self.conv4 = _DoubleConv(in_channels=root_channel*4, out_channels=root_channel*8)

        self.intermedia = _DoubleConv(in_channels=root_channel*8, out_channels=root_channel*8)

        self.conv5 = _DoubleConv(in_channels=root_channel*16, out_channels=root_channel*4)
        self.conv6 = _DoubleConv(in_channels=root_channel*8, out_channels=root_channel*2)
        self.conv7 = _DoubleConv(in_channels=root_channel*4, out_channels=root_channel)
        self.conv8 = _DoubleConv(in_channels=root_channel*2, out_channels=root_channel)

        self.pooling = nn.MaxPool2d(kernel_size=pool_kernel_size)
        # self.upsampling = F.interpolate(x, size=size)
        self.classifier = Conv_Bn_Activation(in_channels=root_channel, out_channels=num_class, activation=None)

    def forward(self, x):
        # TODO: dynamic
        align_corner = False
        low_level = []
        x = self.conv1(x)
        low_level.append(x)
        x = self.pooling(x)

        x = self.conv2(x)
        low_level.append(x)
        x = self.pooling(x)
        
        x = self.conv3(x)
        low_level.append(x)
        x = self.pooling(x)
        
        x = self.conv4(x)
        low_level.append(x)
        x = self.pooling(x)

        x = self.intermedia(x)
        tensor_size = list(x.size())
        x = F.interpolate(x, size=(tensor_size[2]*2, tensor_size[3]*2), mode='bilinear', align_corners=align_corner)

        x = torch.cat([x, low_level.pop()],1)
        x = self.conv5(x)
        tensor_size = list(x.size())
        x = F.interpolate(x, size=(tensor_size[2]*2, tensor_size[3]*2), mode='bilinear', align_corners=align_corner)
        
        x = torch.cat([x, low_level.pop()],1)
        x = self.conv6(x)
        tensor_size = list(x.size())
        x = F.interpolate(x, size=(tensor_size[2]*2, tensor_size[3]*2), mode='bilinear', align_corners=align_corner)

        x = torch.cat([x, low_level.pop()],1)
        x = self.conv7(x)
        tensor_size = list(x.size())
        x = F.interpolate(x, size=(tensor_size[2]*2, tensor_size[3]*2), mode='bilinear', align_corners=align_corner)

        x = torch.cat([x, low_level.pop()],1)
        x = self.conv8(x)
        return nn.Sigmoid()(self.classifier(x))


def create_encoder(in_channels, backbone, pretrained=True, final_flatten=False):
    base_model = creat_torchvision_backbone(in_channels, backbone, pretrained=pretrained, final_flatten=False)
    base_layers = list(base_model.children())[0]
    conv1 = nn.Sequential(*base_layers[:3])
    conv2 = nn.Sequential(*base_layers[3:5])
    conv3 = nn.Sequential(*base_layers[5])
    conv4 = nn.Sequential(*base_layers[6])
    conv5 = nn.Sequential(*base_layers[7])
    encoders = [conv1, conv2, conv3, conv4, conv5]
    return nn.ModuleList(encoders)


def create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, upsample, conv_type):
    # create decoder path consisting of the Decoder modules. The length of the decoder list is equal to `len(f_maps) - 1`
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        if basic_module == DoubleConv:
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        else:
            in_feature_num = reversed_f_maps[i]

        out_feature_num = reversed_f_maps[i + 1]

        # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
        # currently strides with a constant stride: (2, 2, 2)

        _upsample = True
        if i == 0:
            # upsampling can be skipped only for the 1st decoder, afterwards it should always be present
            _upsample = upsample

        decoder = Decoder(in_feature_num, out_feature_num, conv_type,
                          basic_module=basic_module,
                          conv_layer_order=layer_order,
                          conv_kernel_size=conv_kernel_size,
                          num_groups=num_groups,
                          padding=conv_padding,
                          upsample=_upsample)
        decoders.append(decoder)
    return nn.ModuleList(decoders)
