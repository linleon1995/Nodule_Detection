import os
import torch
import torch.nn as nn
import torchvision
from model.model_utils import layers
from model import unet_2d
from model import unet3d
# from model import keras_unet3d
# from model import d2_mask_rcnn
# TODO: Encapsulation with varing first layer, last layer




# def build_keras_unet3d(row, col, index, checkpoint_path):
#     predictor = keras_unet3d.unet_model_3d((1, row, col, index), batch_normalization=True)
#     print(f"[INFO] Load trained model from {checkpoint_path}")
#     predictor.load_weights(checkpoint_path)
#     return predictor


# def build_seg_3d_model()
def build_seg_model(model_name, in_planes, n_class, device, pytorch_pretrained=True, checkpoint_path=None):
    segmnetation_model = select_model(model_name, in_planes, n_class, pytorch_pretrained)
    
    if checkpoint_path is not None:
        if pytorch_pretrained:
            print(f'Warning: the model initial from {checkpoint_path} instead of Pytorch pre-trained model')
        load_model_from_checkpoint(ckpt=checkpoint_path, model=segmnetation_model, device=device)
    return segmnetation_model


def load_model_from_checkpoint(ckpt, model, device):
    common_model_keys = ['model', 'net', 'checkpoint', 'model_state_dict', 'state_dict']
    state_key = torch.load(ckpt, map_location=device)
    model_key = None
    for ckpt_state_key in state_key:
        for test_key in common_model_keys:
            if test_key in ckpt_state_key:
                model_key = ckpt_state_key
                break
    assert model_key is not None, f'The checkpoint model key {model_key} does not exist in self-defined common model key.'
    model.load_state_dict(state_key[model_key])
    model = model.eval()
    model = model.to(device)
    return model


def select_model(model_name, in_planes, n_class, pretrained=True):
    if model_name == '2D-FCN':
        model = torchvision.models.segmentation.fcn_resnet50(pretrained, progress=False)
        model.backbone.conv1 = nn.Conv2d(in_planes, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias = False)
        model.classifier[4] = nn.Conv2d(512, n_class, kernel_size=(1, 1), stride=(1, 1))
    elif model_name == 'DeepLabv3':
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained, progress=False)
        model.backbone.conv1 = nn.Conv2d(in_planes, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias = False)
        model.classifier[4] = nn.Conv2d(256, n_class, kernel_size=(1, 1), stride=(1, 1))
    elif model_name == '2D-Unet':
        model = unet_2d.UNet_2d_backbone(in_channels=in_planes, out_channels=n_class, basic_module=layers.DoubleConv)
    elif model_name == '3D-Unet':
        # TODO: activation
        # TODO: align n_class
        n_class -= 1
        model = unet3d.UNet3D(n_class=n_class)
        # TODO:
        # model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    else:
        raise ValueError(f'Undefined model of {model_name}.')
    return model

