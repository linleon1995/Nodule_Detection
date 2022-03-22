from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from model.model_utils import utils
get_activation = utils.get_activation


# TODO: Define AbstractBackboneBuilder
class PytorchResnetBuilder(nn.Module):
    def __init__(self, in_channels, backbone, pretrained=True, final_flatten=False):
        super(PytorchResnetBuilder, self).__init__()
        self.in_channels = in_channels
        self.backbone = backbone
        self.pretrained = pretrained
        self.final_flatten = final_flatten
        self.model = self.get_model()

    def select_backbone(self):
        if self.backbone == 'resnet18':
            return models.resnet18(pretrained=self.pretrained)
        elif self.backbone == 'resnet34':
            return models.resnet34(pretrained=self.pretrained)
        elif self.backbone == 'resnet50':
            return models.resnet50(pretrained=self.pretrained)
        elif self.backbone == 'resnet101':
            return models.resnet101(pretrained=self.pretrained)
        elif self.backbone == 'resnet152':
            return models.resnet152(pretrained=self.pretrained)
        else:
            raise ValueError('Undefined Backbone Name.')

    def get_model(self):
        # Select backbone
        model  = self.select_backbone()

        # Decide using final classification part (GAP + fc)
        if not self.final_flatten:
            model = nn.Sequential(*list(model.children())[:-2])

        # Modify first convolution layer to accept different input channels
        if self.in_channels != 3:
            conv1 = model[0]
            conv1_out_c = conv1.out_channels
            conv1_ks = conv1.kernel_size
            conv1_stride = conv1.stride
            conv1_padding = conv1.padding
            model[0] = torch.nn.Conv1d(
                self.in_channels, conv1_out_c, conv1_ks, conv1_stride, conv1_padding, bias=False)
        return model

    def forward(self, x):
        return self.model(x)


class PytorchResnextBuilder(PytorchResnetBuilder):
    def __init__(self, in_channels, backbone, pretrained=True, final_flatten=False):
        super(PytorchResnextBuilder, self).__init__(in_channels, backbone, pretrained, final_flatten)

    def select_backbone(self):
        if self.backbone == 'resnext50':
            return models.resnext50_32x4d(pretrained=self.pretrained)
        elif self.backbone == 'resnext101':
            return models.resnext101_32x8d(pretrained=self.pretrained)
        else:
            raise ValueError('Undefined Backbone Name.')


class TimmEfficientNetBuilder(nn.Module):
    def __init__(self, in_channels, backbone, pretrained=True, final_flatten=False):
        super(TimmEfficientNetBuilder, self).__init__()
        self.in_channels = in_channels
        self.backbone = backbone
        self.pretrained = pretrained
        self.final_flatten = final_flatten
        self.model = self.get_model()
        
    def get_model(self):
        return timm.create_model(self.backbone, pretrained=self.pretrained)

    def forward(self, x):
        return self.model(x)


def creat_torchvision_backbone(in_channels, backbone, pretrained, final_flatten=True):
    if 'resnet' in backbone:
        return PytorchResnetBuilder(in_channels, backbone, pretrained, final_flatten)
    elif 'resnext' in backbone:
        return PytorchResnextBuilder(in_channels, backbone, pretrained, final_flatten)


def creat_timm_backbone(in_channels, backbone, pretrained, final_flatten=True):
    if 'efficientnet' in backbone:
        return TimmEfficientNetBuilder(in_channels, backbone, pretrained, final_flatten)