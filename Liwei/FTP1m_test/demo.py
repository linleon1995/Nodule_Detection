'''
Modified Date: 2021/12/14
Author: Li-Wei Hsiao
mail: nfsmw308@gmail.com
'''
import argparse, tqdm, cv2, copy, time, random
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torchvision

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import SimpleITK as sitk
from torch.utils.data import Dataset

class Demo2dSegmentationDataset(Dataset):
    def __init__(self,
                 raw_mhd_path = None, 
                 mask_mhd_path = None,
                 contextSlices_count = 3,
                 contextSlices_shift = 7,
                 fullCt_bool = False,
                 img_size = 512):
        super().__init__()

        self.contextSlices_count = contextSlices_count
        self.contextSlices_shift = contextSlices_shift
        self.layers = self.contextSlices_count * 2 + 1
        self.fullCt_bool = fullCt_bool
        self.img_size = img_size
        self.mask_mhd_path = mask_mhd_path

        ct_mhd = sitk.ReadImage(raw_mhd_path)
        self.ct_hu = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        if mask_mhd_path != None:
            mask_mhd = sitk.ReadImage(mask_mhd_path)
            self.positive_mask = np.array(sitk.GetArrayFromImage(mask_mhd), dtype=np.float32)
            self.positive_indexes = (self.positive_mask.sum(axis=(1,2)).nonzero()[0].tolist())
        else:
            self.positive_mask = None
        
        self.series_uid = raw_mhd_path.split('/')[-1][:-4]

        self.sample_list = []
        if self.fullCt_bool or mask_mhd_path == None:
            self.sample_list += [slice_ndx for slice_ndx in range(self.ct_hu.shape[0]) if slice_ndx%self.contextSlices_shift == 0 or self.contextSlices_shift == 0]
        else:
            self.sample_list += [slice_ndx for slice_ndx in self.positive_indexes if slice_ndx%self.contextSlices_shift == 0]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx):
        slice_ndx = self.sample_list[ndx]
        return self.getitem_fullSlice(slice_ndx)

    def getitem_fullSlice(self, slice_ndx):
        
        ct_a = torch.zeros((self.layers, self.img_size, self.img_size))

        start_ndx = slice_ndx - self.contextSlices_count if slice_ndx - self.contextSlices_count >= 0 else 0
        start_ndx = self.ct_hu.shape[0] - self.layers if start_ndx + self.layers >= self.ct_hu.shape[0] else start_ndx
        
        end_ndx = slice_ndx + self.contextSlices_count + 1 if slice_ndx + self.contextSlices_count + 1 <= self.ct_hu.shape[0] else self.ct_hu.shape[0]
        end_ndx = self.layers if end_ndx < self.layers else end_ndx

        # print(f'{self.series_uid}  slice:{slice_ndx}   ndx:{self.ct_hu.shape[0]}   range:{start_ndx}:{end_ndx}')
        
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, self.ct_hu.shape[0] - 1)
            ct_a[i] = torch.from_numpy(self.ct_hu[context_ndx].astype(np.float32))

        ct_a.clamp_(-1000, 1000)
        
        if self.mask_mhd_path != None:
            pos_a = torch.from_numpy(self.positive_mask[slice_ndx])
        else:
            pos_a = -1

        return ct_a, pos_a, self.series_uid, slice_ndx

def Show(opt, ct_t, masks, preds, series_uid, ct_ndx):
    if masks != []:
        for (ct_img, target, pred, uid, ndx) in zip(ct_t, masks, preds, series_uid, ct_ndx):
            os.makedirs(opt.save_path + '/' + uid, exist_ok=True)
            fig, axs = plt.subplots(1, 3, figsize=(18, 18))
            axs[0].imshow(ct_img[opt.contextSlices_count], cmap='gray')
            axs[1].imshow(ct_img[opt.contextSlices_count], cmap='gray')
            axs[2].imshow(ct_img[opt.contextSlices_count], cmap='gray')
            pred_area = 10
            if pred.sum() > pred_area:
                axs[2].contour(target, 2, cmap='Reds')
                axs[2].contour(pred, 2, cmap='Greens')
            if target.sum() != 0:
                axs[1].contour(target, 2, cmap='Reds')
            axs[0].axis('off')
            axs[1].axis('off')
            axs[2].axis('off')
            axs[0].set_title('raw CT')
            if target.sum() != 0 and pred.sum() > pred_area:
                axs[1].set_title('Ground truth(nodule)')
                axs[2].set_title('Prediction(nodule)')
                plt.savefig(f"{opt.save_path}/{uid}/{ndx}_nodule.png")
                
            elif pred.sum() > pred_area:
                axs[1].set_title('Ground truth')
                axs[2].set_title('Prediction(nodule)')
                plt.savefig(f"{opt.save_path}/{uid}/{ndx}_pred.png")
                
            elif target.sum() != 0:
                axs[1].set_title('Ground truth(nodule)')
                axs[2].set_title('Prediction')
                plt.savefig(f"{opt.save_path}/{uid}/{ndx}_GT.png")
                
            else:
                axs[1].set_title('Ground truth')
                axs[2].set_title('Prediction')
                plt.savefig(f"{opt.save_path}/{uid}/{ndx}.png")
                
            plt.close('all')
    else:
        for (ct_img, pred, uid, ndx) in zip(ct_t, preds, series_uid, ct_ndx):
            os.makedirs(opt.save_path + '/' + uid, exist_ok=True)
            fig, axs = plt.subplots(1, 2, figsize=(18, 18))
            axs[0].imshow(ct_img[opt.contextSlices_count], cmap='gray')
            axs[1].imshow(ct_img[opt.contextSlices_count], cmap='gray')
            pred_area = 10
            if pred.sum() > pred_area:
                axs[1].contour(pred, 2, cmap='Greens')
            axs[0].axis('off')
            axs[1].axis('off')
            axs[0].set_title('raw CT')
            if pred.sum() > pred_area:
                axs[1].set_title('Prediction(nodule)')
                plt.savefig(f"{opt.save_path}/{uid}/{ndx}_pred.png")
            else:
                axs[1].set_title('Prediction')
                plt.savefig(f"{opt.save_path}/{uid}/{ndx}.png")
                
            plt.close('all')

if __name__ in "__main__":
    '''
    1m0002
    ../FTP/malignant_backup/1m0002/1m0002raw mhd/1.2.826.0.1.3680043.2.1125.1.72313825045126389148575820859037795.mhd
    ../FTP/malignant_backup/1m0002/1m0002mask mhd/1.2.826.0.1.3680043.2.1125.1.70155333154775813283332977832847195.mhd
    1m0009
    ../FTP/malignant_backup/1m0009/1m0009raw mhd/1.2.826.0.1.3680043.2.1125.1.68777004745411661442509840697708794.mhd
    ../FTP/malignant_backup/1m0009/1m0009mask mhd/1.2.826.0.1.3680043.2.1125.1.71081228827394195404721867871619159.mhd
    1m0010
    ../FTP/malignant_backup/1m0010/1m0010raw mhd/1.2.826.0.1.3680043.2.1125.1.80971706926297331594083865043075077.mhd
    ../FTP/malignant_backup/1m0010/1m0010mask mhd/1.2.826.0.1.3680043.2.1125.1.19708275299592700952206920455668793.mhd
    1m0015
    ../FTP/malignant_backup/1m0015/1m0015raw mhd/1.2.826.0.1.3680043.2.1125.1.58579217972276171657530170924256606.mhd
    ../FTP/malignant_backup/1m0015/1m0015mask mhd/1.2.826.0.1.3680043.2.1125.1.21303342718931691143151135870146080.mhd
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw-mhd-path',help='path', default='../dataset/FTP/malignant_backup/1m0010/1m0010raw mhd/1.2.826.0.1.3680043.2.1125.1.80971706926297331594083865043075077.mhd')
    parser.add_argument('--mask-mhd-path',help='path or None', default='../dataset/FTP/malignant_backup/1m0010/1m0010mask mhd/1.2.826.0.1.3680043.2.1125.1.19708275299592700952206920455668793.mhd')

    parser.add_argument('--batch-size', type=int, default=1, help='Batch size to use for training')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of worker processes for background data loading')
    
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--contextSlices_count', type=int, default=3)
    parser.add_argument('--contextSlices_shift', type=int, default=1)
    parser.add_argument('--fullCt_bool', type=bool, default=False)
    parser.add_argument('--image_size', type=int, default=512)

    parser.add_argument('--pretrain_path', default='./model/FCN_all_best.pt')
    parser.add_argument('--save_path', default='./output')
    parser.add_argument('--save', default=True)
    opt = parser.parse_args()
    
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

    print('Load data......')
    total_sec = 0
    total_slice = 0
    start = time.time()
    test_ds = Demo2dSegmentationDataset(
                raw_mhd_path = opt.raw_mhd_path, 
                mask_mhd_path = opt.mask_mhd_path,
                contextSlices_count=opt.contextSlices_count,
                contextSlices_shift=opt.contextSlices_shift,
                fullCt_bool=opt.fullCt_bool,
                img_size=opt.image_size,
            )
    test_dl = DataLoader( test_ds, batch_size=opt.batch_size, num_workers=opt.num_workers)
    total_sec = time.time() - start

    print('Demo...')
    total_slice_sec = 0
    segmentation_model.eval()
    start = time.time()
    with torch.no_grad():
        for idx, (ct_t, pos_t, series_uid, ct_ndx) in enumerate(test_dl):
            if opt.mask_mhd_path != None:
                ct_g, pos_g = ct_t.to(device, non_blocking=True), pos_t.to(device, non_blocking=True)
                masks = pos_g.cpu().data.numpy()
            else:
                ct_g = ct_t.to(device, non_blocking=True)

            
            result = segmentation_model(ct_g)['out']
            result = nn.Softmax(dim=1)(result)
            preds = torch.argmax(result, dim=1)
            total_slice += ct_t.size(0)
            preds = preds.cpu().data.numpy()

            if opt.save:
                if opt.mask_mhd_path != None:
                    Show(opt, ct_t, masks, preds, series_uid, ct_ndx)
                else:
                    Show(opt, ct_t, [], preds, series_uid, ct_ndx)
    total_sec += (time.time() - start)
    total_slice_sec +=(time.time() - start)
    print('Done. Slices = { %d slice } Computing time = {Total: %f s } {Mean 1 Dataloader+slice: %f s} {Mean 1 slice: %f s}'%(total_slice, total_sec, total_sec / total_slice, total_slice_sec / total_slice))