import torch
import torch.nn as nn
import torchvision

# TODO: Encapsulation with varing first layer, last layer


def build_seg_model(model_name, slice_shift, n_class, device, pytorch_pretrained=True, checkpoint_path=None):
    in_planes = 2*slice_shift + 1
    segmnetation_model = select_model(model_name, in_planes, n_class, pytorch_pretrained)
    
    if checkpoint_path is not None:
        if pytorch_pretrained:
            print(f'Warning: the model initial from {checkpoint_path} instead of Pytorch pre-trained model')
        load_model_from_checkpoint(ckpt=checkpoint_path, model=segmnetation_model, device=device)
    return segmnetation_model


def load_model_from_checkpoint(ckpt, model, device):
    common_model_keys = ['model', 'net', 'checkpoint']
    state_key = torch.load(ckpt, map_location=device)
    for k in state_key:
        for model_key in common_model_keys:
            if model_key in k:
                model_key = k
                break
    model.load_state_dict(state_key[model_key])
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
        pass
    else:
        raise ValueError(f'Undefined model of {model_name}.')
    return model

