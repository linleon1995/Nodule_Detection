
from turtle import forward
import torch
import torch.nn as nn
import torchvision

from modules.utils import configuration


# def load_model_from_checkpoint(ckpt, model, device, model_key):
#     state_key = torch.load(ckpt, map_location=device)
#     model.load_state_dict(state_key[model_key])
#     model = model.to(device)
#     return model

# TODO: raise if input undefined model

# TODO: merge pretrained and checkpoint path
def build_model(model_name, slice_shift, n_class, pretrained=True, checkpoint_path=None, model_key='net'):
    in_planes = 2*slice_shift + 1
    segmnetation_model = SegModel(
        model_name, in_planes, n_class, pretrained, device=configuration.get_device(), model_key=model_key, checkpoint_path=checkpoint_path)
    # if checkpoint_path is not None:
    #     load_model_from_checkpoint(ckpt=checkpoint_path, model=segmnetation_model, device=configuration.get_device(), model_key=model_key)
    return segmnetation_model


class SegModel(nn.Module):
    def __init__(self, model_name, in_planes, n_class, pretrained, device, model_key, checkpoint_path=None):
        super().__init__()
        self.model = self.build_model(model_name, in_planes, n_class, pretrained)
        if checkpoint_path is not None:
            self.model = self.load_model_from_checkpoint(checkpoint_path, self.model, device, model_key)

    def build_model(self, model_name, in_planes, n_class, pretrained=True):
        if model_name == '2D-FCN':
            model = torchvision.models.segmentation.fcn_resnet50(pretrained, progress=False)
            model.backbone.conv1 = nn.Conv2d(in_planes, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias = False)
            model.classifier[4] = nn.Conv2d(512, n_class, kernel_size=(1, 1), stride=(1, 1))
        elif model_name == 'DeepLabv3':
            model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained, progress=False)
            model.backbone.conv1 = nn.Conv2d(in_planes, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias = False)
            model.classifier[4] = nn.Conv2d(256, n_class, kernel_size=(1, 1), stride=(1, 1))
        return model

    def load_model_from_checkpoint(self, checkpoint_path, model, device, model_key):
        state_key = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_key[model_key])
        model = model.to(device)
        return model

    def forward(self, x):
        return self.model(x)
