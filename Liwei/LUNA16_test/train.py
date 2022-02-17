'''
Modified Date: 2021/12/14
Author: Li-Wei Hsiao
mail: nfsmw308@gmail.com

pip install diskcache==4.1
'''

import argparse
import os, random, cv2, copy, tqdm, datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import torchvision

from dataset_seg import Luna2dSegmentationDataset, TrainingLuna2dSegmentationDataset
from logconf import logging
from model import FocalLoss, SoftIoULoss, BinaryDiceLoss, SoftDiceLoss

def random_fixed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class metrics:
    def __init__(self, n_class):
        self.n_class = n_class if n_class != 1 else 2
        self.class_intersection = np.zeros(self.n_class)
        self.class_union = np.zeros(self.n_class)
        self.class_gt_area = np.zeros(self.n_class)
        self.class_seg_area = np.zeros(self.n_class)
        # metrics
        self.class_acc = np.zeros(self.n_class)
        self.class_iou = np.zeros(self.n_class)
        self.class_f1 = np.zeros(self.n_class)
        self.pixel_TP = 0
        self.pixel_FP = 0
        self.pixel_FN = 0

    def calculate(self, predicts, targets):
        if targets.ndim == 4:
            gts = np.zeros((targets.shape[0], targets.shape[2], targets.shape[3]))
            for i in range(0, targets.shape[1]):
                gts[targets[:, i, :, :] == i] = i
            gts = gts.astype('int')
            targets = copy.deepcopy(gts)
        for seg, gt in zip(predicts, targets):
            for j in range(1, self.n_class):
                self.class_intersection[j] += np.logical_and(gt==j,seg==j).sum()
                self.class_gt_area[j] += (gt==j).sum()
                self.class_seg_area[j] += (seg==j).sum()
                
            self.pixel_TP += np.logical_and(gt, seg).sum()
            self.pixel_FP += np.logical_and(np.logical_xor(gt, seg), seg > 0).sum()
            self.pixel_FN += np.logical_and(np.logical_xor(gt, seg), gt > 0).sum()

    def evaluation(self, show=False):
        for k in range(self.n_class):
            self.class_acc[k] = self.class_intersection[k]/self.class_gt_area[k] if self.class_gt_area[k] != 0 else 0
            self.class_iou[k] = self.class_intersection[k]/(self.class_gt_area[k]+self.class_seg_area[k]-self.class_intersection[k]) if (self.class_gt_area[k]+self.class_seg_area[k]-self.class_intersection[k]) != 0 else 0
            if (self.class_gt_area[k]+self.class_seg_area[k]) == 0 and 2*self.class_intersection[k] == 0:
                self.class_f1[k] = 1 if k != 0 else 0
            else:
                self.class_f1[k] = 2*self.class_intersection[k]/(self.class_gt_area[k]+self.class_seg_area[k])
       
        mIOU = self.class_iou[1:].mean()
        Total_dice = 2*self.class_intersection[1:].sum()/(self.class_gt_area[1:].sum()+self.class_seg_area[1:].sum()) if (self.class_gt_area[1:].sum()+self.class_seg_area[1:].sum()) != 0 else 0
        pixel_Precision = self.pixel_TP / (self.pixel_TP + self.pixel_FP) if (self.pixel_TP + self.pixel_FP) != 0 else 0
        pixel_Recall = self.pixel_TP / (self.pixel_TP + self.pixel_FN) if (self.pixel_TP + self.pixel_FN) != 0 else 0
        pixel_F1_score = 2*(pixel_Precision*pixel_Recall / (pixel_Recall + pixel_Precision)) if (pixel_Recall + pixel_Precision) != 0 else 0

        if show:
            print('class accuracy =',self.class_acc)
            print('class IoU =',self.class_iou)
            print('class dice =',self.class_f1)
            print('Pixel Precision =',pixel_Precision)
            print('Pixel Recall =',pixel_Recall)
            print('Pixel F1 score =',pixel_F1_score)
            print('mIoU =', mIOU)
            print('total dice =', Total_dice)
        return self.class_acc, self.class_iou, self.class_f1, mIOU, pixel_Precision, pixel_Recall, Total_dice

if __name__ in "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size to use for training')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of worker processes for background data loading')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs to train for')
    parser.add_argument('--start_valid_epoch', type=int, default=1)
    parser.add_argument('--validation_cadence', type=int, default=1)
    
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--val_stride', type=int, default=10)
    
    parser.add_argument('--train_contextSlices_count', type=int, default=3)
    parser.add_argument('--train_contextSlices_shift', type=int, default=1)
    parser.add_argument('--train_fullCt_bool', type=bool, default=False)
    
    parser.add_argument('--valid_contextSlices_count', type=int, default=3)
    parser.add_argument('--valid_contextSlices_shift', type=int, default=1)
    parser.add_argument('--valid_fullCt_bool', type=bool, default=False)

    parser.add_argument('--adam', type=bool, default = True, help='use torch.optim.Adam() or torch.optim.SGD() optimizer')
    parser.add_argument('--lr', type=float, default = 1e-4, help='optimizer learning rate')
    parser.add_argument('--momentum', type=float, default = 0.9, help='optimizer momentum')
    parser.add_argument('--weight_decay', type=float, default = 5e-4, help='optimizer weight_decay')
    parser.add_argument('--lr_decay', type=list, default = [20,40], help='When does the epoch decrease the learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default = 0.1, help='decrease rate')
    
    parser.add_argument('--classification_Threshold', type=float, default = 0.5)

    parser.add_argument('--loss', type=str, default='SoftIoU', help='use BinaryDice or Focal or SoftIoU or CrossEntropy')
    parser.add_argument('--gamma', type=float, default=2, help='Focal loss gamma')
    parser.add_argument('--model_name', default='FCN', help="FCN or DeeplabV3.")
    parser.add_argument('--save_chackpoint', default='IOUFocal', help="Save chackpoint folder name")
    parser.add_argument('--pretrain_weight', default=False, help="False is not pretrain weight, Using pretrained weight the path ex. ./FCN_IOUFocal")
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
    random_fixed(2021)
    os.makedirs(f"{opt.model_name}_{opt.save_chackpoint}", exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print('Load model parameters......')
    if opt.model_name == "FCN":
        segmentation_model = torchvision.models.segmentation.fcn_resnet50(pretrained=True, progress=False)
        segmentation_model.backbone.conv1 = nn.Conv2d(opt.train_contextSlices_count*2 + 1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias = False)
        segmentation_model.classifier[4] = nn.Conv2d(512, opt.n_classes, kernel_size=(1, 1), stride=(1, 1))
    else:
        segmentation_model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True, progress=False)
        segmentation_model.backbone.conv1 = nn.Conv2d(opt.train_contextSlices_count*2 + 1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias = False)
        segmentation_model.classifier[4] = nn.Conv2d(256, opt.n_classes, kernel_size=(1, 1), stride=(1, 1))

    print(segmentation_model)

    if opt.pretrain_weight:
        segmentation_model.load_state_dict(torch.load(opt.pretrain_weight)['model_state_dict'])
    
    if use_cuda:
        segmentation_model = segmentation_model.to(device)

    if opt.adam:
        optimizer = torch.optim.Adam(segmentation_model.parameters(), lr = opt.lr)
    else:
        optimizer = torch.optim.SGD(segmentation_model.parameters(), lr = opt.lr, momentum = opt.momentum, weight_decay = opt.weight_decay)

    if opt.loss == 'BinaryDice':
        criterion = BinaryDiceLoss()
        criterionFN = BinaryDiceLoss()
    elif opt.loss == 'SoftIoU':
        criterion = SoftIoULoss(opt.n_classes)
        criterionFocal = FocalLoss(gamma = opt.gamma)
    elif opt.loss == "Focal":
        criterion = FocalLoss(gamma = opt.gamma)
    else:
        criterion = nn.CrossEntropyLoss()

    print('Load Dataset......')
    train_ds = TrainingLuna2dSegmentationDataset(
                val_stride=opt.val_stride,
                isValSet_bool=False,
                contextSlices_count=opt.train_contextSlices_count,
                contextSlices_shift=opt.train_contextSlices_shift,
                fullCt_bool=opt.train_fullCt_bool,
                shift = 64,
                n_class = opt.n_classes,
            )
    train_dl = DataLoader( train_ds, batch_size = opt.batch_size, num_workers = opt.num_workers, pin_memory = use_cuda)
    
    val_ds = Luna2dSegmentationDataset(
                val_stride=opt.val_stride,
                isValSet_bool=True,
                contextSlices_count=opt.valid_contextSlices_count,
                contextSlices_shift=opt.valid_contextSlices_shift,
                fullCt_bool=opt.valid_fullCt_bool,
            )
    val_dl = DataLoader( val_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=use_cuda)

    print('Training......')
    best_mIOU = 0.0
    best_r = 0
    best_p = 0
    best_dice = 0.0
    for epoch in range(1, opt.epochs + 1):
        print(f"Starting epoch {epoch} of {opt.epochs}")
        segmentation_model.train()
        train_dl.dataset.shuffleSamples()
        if epoch in opt.lr_decay:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * opt.lr_decay_rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']

        training_loss = 0
        for idx, (input_t, label_t, label_ts, series_list, _slice_ndx_list) in enumerate(train_dl):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not 'BinaryDice' in opt.loss:
                label_t = label_t.long()
            input_g, label_g, label_gs = input_t.to(device, non_blocking=True), label_t.to(device, non_blocking=True), label_ts.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            prediction_g = segmentation_model(input_g)['out']

            if opt.loss == 'BinaryDice':
                prediction_g = prediction_g.squeeze()
                if prediction_g.size() != label_g.size():
                    prediction_g = prediction_g.unsqueeze(0)
                loss = criterion(prediction_g, label_g) + criterionFN(prediction_g * label_g, label_g) * 8
                loss.backward()
                optimizer.step()
            else:
                loss = criterion(prediction_g, label_g) + criterionFocal(prediction_g, label_g)
                loss.backward()
                optimizer.step()
            
            training_loss += loss.item()
            if idx % int(len(train_dl)/5) == 0 or idx ==len(train_dl):
                print("%s        [Batch %d / %d] lr: %0.6f | Train loss: %4.5f"%(str(datetime.datetime.today()), idx, len(train_dl), learning_rate, training_loss))
        
        print( "%s[Train]  Epoch: %d/%d  |"%(str(datetime.datetime.today()), epoch, opt.epochs) + \
                   "lr: %0.6f  |"%(learning_rate) + \
                   "Train loss: %4.5f"%(training_loss))

        if epoch == 1 or (epoch >= opt.start_valid_epoch and epoch % opt.validation_cadence == 0):
            metric = metrics(opt.n_classes)
            segmentation_model.eval()
            with torch.no_grad():
                dice = 0
                pixel_accuracy = 0
                total = 0
                for idx, (input_t, label_t, series_list, _slice_ndx_list) in enumerate(tqdm.tqdm(val_dl)):
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    if not 'BinaryDice' in opt.loss:
                        label_t = label_t.long()
                    input_g, label_g = input_t.to(device, non_blocking=True), label_t.to(device, non_blocking=True)
            
                    result = segmentation_model(input_g)['out']
                    if not 'BinaryDice' in opt.loss:
                        result = nn.Softmax(dim=1)(result)
                        preds = torch.argmax(result, dim=1)
                    else:
                        result = result.squeeze()
                        if result.size() != label_g.size():
                            result = result.unsqueeze(0)
                        preds = (result > opt.classification_Threshold).to(torch.float32)
                    preds = preds.cpu().data.numpy()
                    masks = label_g.cpu().data.numpy()
                    metric.calculate(preds, masks)
                class_acc, class_iou, class_f1, mIOU, pixel_Precision, pixel_Recall, Total_dice = metric.evaluation()
                dict_models = {'epoch': epoch,
                                'model_state_dict': segmentation_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lr': learning_rate,
                                'dice': Total_dice
                                }
                best_r = max(best_r, pixel_Recall)
                best_p = max(best_p, pixel_Precision)
                best_mIOU = max(best_mIOU, mIOU)
                
                if best_dice <= Total_dice:
                    torch.save(dict_models, f"{opt.model_name}_{opt.save_chackpoint}/best.pt")
                    best_dice = Total_dice
                    print("save best model!")
                torch.save(dict_models, f"{opt.model_name}_{opt.save_chackpoint}/last.pt")
                print('{:s}[Valid]  Dice: {:7.5f} mIOU: {:7.5f} Best dice: {:7.5f} Best mIOU: {:7.5f} P: {:7.5f} R: {:7.5f}'.format(str(datetime.datetime.today()), Total_dice, mIOU, best_dice, best_mIOU, pixel_Precision, pixel_Recall ))

