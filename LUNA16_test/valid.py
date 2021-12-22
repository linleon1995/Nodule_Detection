'''
Modified Date: 2021/12/14
Author: Li-Wei Hsiao
mail: nfsmw308@gmail.com
'''
import argparse, tqdm, cv2, time, copy
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torchvision

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset_seg import Luna2dSegmentationDataset
from util import metrics

if __name__ in "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size to use for training')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of worker processes for background data loading')
    
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--val_stride', type=int, default=10)
    parser.add_argument('--contextSlices_count', type=int, default=3)
    parser.add_argument('--contextSlices_shift', type=int, default=1)
    parser.add_argument('--fullCt_bool', type=bool, default=True)
        
    parser.add_argument('--loss', type=str, default='SoftIoU', help='use BinaryDice or Focal or SoftIoU or CrossEntropy')
    parser.add_argument('--pretrain_path', default='./FCN_IOUFacal/best.pt')
    parser.add_argument('--save_path', default='./show_valid')
    parser.add_argument('--save', default=False)
    opt = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if opt.save:
        os.makedirs(opt.save_path, exist_ok=True)

    print('Load model......')
    if "FCN" in opt.pretrain_path:
        segmentation_model = torchvision.models.segmentation.fcn_resnet50(pretrained=True, progress=False)
        segmentation_model.backbone.conv1 = nn.Conv2d(opt.contextSlices_count*2 + 1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias = False)
        segmentation_model.classifier[4] = nn.Conv2d(512, opt.n_classes, kernel_size=(1, 1), stride=(1, 1))
    else:
        segmentation_model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True, progress=False)
        segmentation_model.backbone.conv1 = nn.Conv2d(opt.train_contextSlices_count*2 + 1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias = False)
        segmentation_model.classifier[4] = nn.Conv2d(256, opt.n_classes, kernel_size=(1, 1), stride=(1, 1))

    if use_cuda:
        segmentation_model = segmentation_model.to(device)
    
    segmentation_model.load_state_dict(torch.load(opt.pretrain_path)['model_state_dict'])

    print('Load validation set......')
    val_ds = Luna2dSegmentationDataset(
                val_stride=opt.val_stride,
                isValSet_bool=True,
                contextSlices_count=opt.contextSlices_count,
                contextSlices_shift=opt.contextSlices_shift,
                fullCt_bool=opt.fullCt_bool,
            )
    val_dl = DataLoader( val_ds, batch_size=opt.batch_size, num_workers=opt.num_workers)

    print('valid...')
    metric = metrics(opt.n_classes)
    segmentation_model.eval()
    with torch.no_grad():
        dice = 0
        pixel_accuracy = 0
        total = 0
        for series_ndx, (ct_t, pos_t, series_uid, ct_ndx) in enumerate(tqdm.tqdm(val_dl)):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not 'BinaryDice' in opt.loss:
                pos_t = pos_t.long()
            ct_g, pos_g = ct_t.to(device, non_blocking=True), pos_t.to(device, non_blocking=True)
            # start = time.time()
            result = segmentation_model(ct_g)['out']
            result = nn.Softmax(dim=1)(result)
            preds = torch.argmax(result, dim=1)
            # print(time.time() - start)
            preds = preds.cpu().data.numpy()
            masks = pos_g.cpu().data.numpy()
            metric.calculate(preds, masks)
            if opt.save:
                for (ct_img, target, pred, uid, ndx) in zip(ct_t, masks, preds, series_uid, ct_ndx):
                    os.makedirs(opt.save_path + '/' + uid, exist_ok=True)
                    plt.imshow(ct_img[opt.contextSlices_count], cmap='gray')
                    if pred.sum() != 0:
                        plt.contour(pred, 4, cmap='Greens')
                    if target.sum() != 0:
                        plt.contour(target, 4, cmap='Reds')
                    plt.axis('off')
                    if target.sum() != 0 and pred.sum() != 0:
                        plt.savefig(f"{opt.save_path}/{uid}/{ndx}_nodule.png")
                    else:
                        plt.savefig(f"{opt.save_path}/{uid}/{ndx}.png")
                    # plt.show()
                    plt.close('all')
        class_acc, class_iou, class_f1, mIOU, pixel_Precision, pixel_Recall, Total_dice = metric.evaluation(True)