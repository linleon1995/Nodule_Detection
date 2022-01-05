import argparse, tqdm, cv2, copy, time
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torchvision

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset_seg import TestingLuna2dSegmentationDataset
from util import metrics

if __name__ in "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size to use for training')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of worker processes for background data loading')
    
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--contextSlices_count', type=int, default=3)
    parser.add_argument('--contextSlices_shift', type=int, default=1)
    parser.add_argument('--fullCt_bool', type=bool, default=True)
        
    parser.add_argument('--loss', type=str, default='SoftIoU', help='use BinaryDice or Focal or SoftIoU or CrossEntropy')
    parser.add_argument('--pretrain_path', default=rf'C:\Users\test\Desktop\Leon\Projects\Nodule_project_LiweiHsiao\FCN_IOUFocal/best.pt')
    parser.add_argument('--save_path', default='./plot')
    parser.add_argument('--save', default=False)
    parser.add_argument('--series_uid', default=None)
    opt = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
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

    # +++
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    import utils
    from volume_eval import volumetric_data_eval

    # check_point_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_003'
    # check_point_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_010'
    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.DATALOADER.NUM_WORKERS = 0
    # # cfg.MODEL.WEIGHTS = os.path.join(check_point_path, "model_final.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(check_point_path, "model_0039999.pth")  # path to the model we just trained
    # # cfg.MODEL.WEIGHTS = os.path.join(check_point_path, "model_0069999.pth")  # path to the model we just trained
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # cfg.INPUT.MASK_FORMAT = 'bitmask'
    # cfg.OUTPUT_DIR = check_point_path
    # cfg.DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI'
    # # cfg.INPUT.MIN_SIZE_TEST = 0
    # # cfg.INPUT.MAX_SIZE_TEST = 480

    # run = os.path.split(check_point_path)[1]
    # weight = os.path.split(cfg.MODEL.WEIGHTS)[1].split('.')[0]
    # cfg.SAVE_PATH = rf'C:\Users\test\Desktop\Leon\Weekly\1217\maskrcnn-{run}-{weight}-samples'
    # # TODO: dataset path in configuration
    # cfg.FULL_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-all-slices'
    # # cfg.FULL_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant'
    # # cfg.FULL_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16'
    # cfg.SAVE_COMPARE = True
    # # cfg.CASE_INDICES = list(range(10))
    # cfg.CASE_INDICES = list(range(10, 20))
    # # cfg.CASE_INDICES = list(range(810, 820))
    # cfg.ONLY_NODULES = True
    # # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    # segmentation_model = DefaultPredictor(cfg)

    # def get_pred(outputs):
    #     pred = outputs["instances"]._fields['pred_masks'].cpu().detach().numpy() 
    #     pred = np.sum(pred, axis=0)
    #     pred = np.where(pred>=1, 1, 0)
    #     pred = pred[np.newaxis]
    #     pred = torch.from_numpy(pred)
    #     return pred

    # def preprocess(img):
    #     img = img.cpu().data.numpy()
    #     return utils.raw_preprocess(img, lung_segment=True, norm=True, change_channel=True, output_dtype=np.uint8)
    vol_metric = volumetric_data_eval()
    vol_idx = 0
    # +++

    print('Load Testing set......')
    test_ds = TestingLuna2dSegmentationDataset(
                series_uid=opt.series_uid,
                contextSlices_count=opt.contextSlices_count,
                contextSlices_shift=opt.contextSlices_shift,
                fullCt_bool=opt.fullCt_bool,
                img_size = 512,
            )
    test_dl = DataLoader( test_ds, batch_size=opt.batch_size, num_workers=opt.num_workers)

    print('Testing...')
    metric = metrics(opt.n_classes)
    segmentation_model.eval()
    with torch.no_grad():
        dice = 0
        pixel_accuracy = 0
        total = 0
        for series_ndx, (ct_t, pos_t, series_uid, ct_ndx) in enumerate(tqdm.tqdm(test_dl)):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not 'BinaryDice' in opt.loss:
                pos_t = pos_t.long()
            ct_g, pos_g = ct_t.to(device, non_blocking=True), pos_t.to(device, non_blocking=True)
            
            # +++
            result = segmentation_model(ct_g)['out']
            result = nn.Softmax(dim=1)(result)
            preds = torch.argmax(result, dim=1)
            # img = preprocess(ct_g[0,0])
            # outputs = segmentation_model(img)
            # preds = get_pred(outputs)
            # +++
            
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
                    plt.close('all')
                    # plt.show()

        # +++
            if series_ndx == 0:
                last_series_uid = series_uid[0]
                mask_vol = masks
                pred_vol = preds

            if series_uid[0] == last_series_uid:
                mask_vol = np.concatenate([mask_vol, masks], axis=0)
                pred_vol = np.concatenate([pred_vol, preds], axis=0)
            else:
                vol_metric.calculate(mask_vol, pred_vol)
                mask_vol = masks
                pred_vol = preds
            
            last_series_uid = series_uid[0]


        nodule_tp, nodule_fp, nodule_fn, nodule_precision, nodule_recall = vol_metric.evaluation(show_evaluation=True)
        # +++
        class_acc, class_iou, class_f1, mIOU, pixel_Precision, pixel_Recall, Total_dice = metric.evaluation(True)