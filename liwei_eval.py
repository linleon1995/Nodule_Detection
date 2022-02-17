import argparse, tqdm
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torchvision

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Liwei.FTP1m_test import test

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=4, help='Batch size to use for training')
parser.add_argument('--num-workers', type=int, default=0, help='Number of worker processes for background data loading')

parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--val_stride', type=int, default=10)
parser.add_argument('--contextSlices_count', type=int, default=3)
parser.add_argument('--contextSlices_shift', type=int, default=1)
parser.add_argument('--fullCt_bool', type=bool, default=True)
    
parser.add_argument('--loss', type=str, default='SoftIoU', help='use BinaryDice or Focal or SoftIoU or CrossEntropy')
parser.add_argument('--pretrain_path', default=rf'C:\Users\test\Desktop\Leon\Projects\Nodule_project_LiweiHsiao\FCN_IOUFocal/best.pt')
parser.add_argument('--save_path', default='./Show_output')
parser.add_argument('--save', default=False)
opt = parser.parse_args()

def liwei_predictor(opt, images):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if opt.save:
        os.makedirs(opt.save_path, exist_ok=True)

    print('Load model......')
    if 'FCN' in opt.pretrain_path:
        segmentation_model = torchvision.models.segmentation.fcn_resnet50(pretrained=True, progress=False)
        segmentation_model.backbone.conv1 = nn.Conv2d(opt.contextSlices_count*2 + 1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias = False)
        segmentation_model.classifier[4] = nn.Conv2d(512, opt.n_classes, kernel_size=(1, 1), stride=(1, 1))
    else:
        segmentation_model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True, progress=False)
        segmentation_model.backbone.conv1 = nn.Conv2d(opt.contextSlices_count*2 + 1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias = False)
        segmentation_model.classifier[4] = nn.Conv2d(256, opt.n_classes, kernel_size=(1, 1), stride=(1, 1))

    if use_cuda:
        segmentation_model = segmentation_model.to(device)
    
    segmentation_model.load_state_dict(torch.load(opt.pretrain_path)['model_state_dict'])

    print('test...')
    segmentation_model.eval()
    with torch.no_grad():
        images = [image.astype("float32").transpose(0, 3, 1, 2) for image in images]
        images = torch.from_numpy(np.concatenate(images, axis=0))
        result = segmentation_model(images)
        logits = result['out']
        logits = nn.Softmax(dim=1)(logits)
        preds = torch.argmax(logits, dim=1)
        preds = preds.cpu().data.numpy()
    return preds



def liwei_eval():
    volume_generator = test.asus_pred_generator

def main(opt):
    preds = liwei_predictor(opt, images)


if __name__ == '__main__':
    main(opt)