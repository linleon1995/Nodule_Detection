
from turtle import forward
import torch
import torch.nn as nn
import torchvision


def build_model(model_name, slice_shift, n_class, pretrained=True):
    in_planes = 2*slice_shift + 1
    segmnetation_model = SegModel(model_name, in_planes, n_class, pretrained)
    return segmnetation_model


class SegModel(nn.Module):
    def __init__(self, model_name, in_planes, n_class, pretrained):
        self.model = self.build_model(model_name, in_planes, n_class, pretrained)

    @classmethod
    def build_model(cls, model_name, in_planes, n_class, pretrained=True):
        if model_name == '2D-FCN':
            model = torchvision.models.segmentation.fcn_resnet50(pretrained, progress=False)
            model.backbone.conv1 = nn.Conv2d(in_planes, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias = False)
            model.classifier[4] = nn.Conv2d(512, n_class, kernel_size=(1, 1), stride=(1, 1))
        elif model_name == 'DeepLabv3':
            model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained, progress=False)
            model.backbone.conv1 = nn.Conv2d(in_planes, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias = False)
            model.classifier[4] = nn.Conv2d(256, n_class, kernel_size=(1, 1), stride=(1, 1))
        return model

    def forward(self, x):
        return self.model(x)
