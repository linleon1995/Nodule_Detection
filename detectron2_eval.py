from detectron2.evaluation import COCOEvaluator, inference_on_dataset, SemSegEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import save
from pylidc.utils import consensus
from pathlib import Path
from statistics import median_high
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import cv2_imshow, calculate_malignancy, segment_lung
from convert_to_coco_structure import lidc_to_datacatlog_valid
import logging
from sklearn.metrics import confusion_matrix
import time
import pylidc as pl
import pandas as pd
logging.basicConfig(level=logging.INFO)

import site_path
from modules.data import dataset_utils
from modules.utils import metrics
from modules.utils import metrics2
from modules.utils import evaluator


def eval(cfg):
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    predictor = DefaultPredictor(cfg)

    register_coco_instances("my_dataset_valid", {}, "annotations_valid.json", rf"C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-Preprocessing-png\Image")
    # DatasetCatalog.register("my_dataset_valid", lidc_to_datacatlog_valid)
    dataset_dicts = DatasetCatalog.get("my_dataset_valid")
    metadata = MetadataCatalog.get("my_dataset_valid")

    for d in random.sample(dataset_dicts, 3):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=metadata, 
                    scale=1.0, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2_imshow(out.get_image()[:, :, ::-1])


    evaluator = COCOEvaluator("my_dataset_valid", cfg, distributed=False, output_dir="./output")
    # evaluator = SemSegEvaluator("my_dataset_valid", distributed=False, num_classes=2, output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "my_dataset_valid")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`


def volume_eval3(cfg, case_list):
    # This is to name each image and mask
    prefix = [str(x).zfill(3) for x in range(1000)]
    mask_threshold = 8

    # lidc_evaluator = evaluator.ClassificationEvaluator(num_class=cfg.MODEL.ROI_HEADS.NUM_CLASSES+1)
    # lidc_evaluator.register_new_metrics({"DSC": metrics2.mean_dsc, 'IoU': metrics2.mean_iou})
    predictor = DefaultPredictor(cfg)
    total_dscs, total_ious = None, None
    for patient in case_list:
        pid = patient #LIDC-IDRI-0001~
        # +++
        scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid)
        num_scan_in_one_patient = scans.count()
        print(f'{pid} has {num_scan_in_one_patient} scan')
        scan_list = scans.all()
        for scan_idx, scan in enumerate(scan_list):
            # scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            # +++
            if scan is None:
                print(scan)
            nodules_annotation = scan.cluster_annotations()
            vol = scan.to_volume()
            
            print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid, vol.shape, len(nodules_annotation)))

            # Patients with nodules
            masks_vol = np.zeros_like(vol)
            for nodule_idx, nodule in enumerate(nodules_annotation):
            # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
            # This current for loop iterates over total number of nodules in a single patient
                mask, cbbox, masks = consensus(nodule, clevel=0.5, pad=512)
                malignancy, cancer_label = calculate_malignancy(nodule)
                if malignancy >= 3:
                    cancer_categories = 2
                else:
                    cancer_categories = 1

                masks_vol += cancer_categories*mask

                # lung_np_array = vol[cbbox]
                # We calculate the malignancy information

            assert np.shape(vol) == np.shape(masks_vol)
            pred_vol = np.zeros_like(vol)
            for img_idx in range(vol.shape[2]):
                img = segment_lung(vol[...,img_idx])
                img[img==-0] = 0
                img = np.uint8(255*((img-np.min(img))/(1e-7+np.max(img)-np.min(img))))

                # TODO:
                img = np.tile(img[...,np.newaxis], (1,1,3))
                outputs = predictor(img) 
                pred_mask = outputs["instances"]._fields['pred_masks'].cpu().detach().numpy() 
                pred_classes = outputs["instances"]._fields['pred_classes'].cpu().detach().numpy() 
                pred_classes = np.reshape(pred_classes, (pred_classes.shape[0], 1, 1))
                pred_classes += 1
                # pred_mask = np.sum(pred_mask, axis=0)
                pred_mask = np.sum(pred_mask*pred_classes, axis=0)
                pred_mask = np.int32(pred_mask)
                
                pred_vol[...,img_idx] = pred_mask
                # print('idx, gt, pred', img_idx, np.sum(masks_vol[...,img_idx]), np.sum(pred_vol[...,img_idx]))
                # fig, ax = compare_result(img, masks_vol[...,img_idx], pred_vol[...,img_idx], alpha=0.2)
                # fig.savefig(os.path.join(cfg.SAVE_PATH, f'{pid}-{scan_idx}-{img_idx}.png'), figsize=(10,10))
                # if img_idx > 5:
                #     break

            cm = confusion_matrix(np.reshape(masks_vol, [-1]), np.reshape(pred_vol, [-1]), labels=np.arange(0, cfg.MODEL.ROI_HEADS.NUM_CLASSES+1))
            mean_dsc, dscs = metrics2.mean_dsc(cm)
            mean_iou, ious = metrics2.mean_iou(cm)
            # 64 65 66
            # fig, ax = plt.subplots(1,2)
            # ax[0].imshow(masks_vol[...,242])
            # ax[1].imshow(pred_vol[...,242])
            # plt.savefig('xx.png')
            if total_dscs is None:
                total_dscs = dscs[np.newaxis]
                total_ious = ious[np.newaxis]
            else:
                total_dscs = np.append(total_dscs, dscs[np.newaxis], axis=0)
                total_ious = np.append(total_ious, ious[np.newaxis], axis=0)
            # if np.sum(masks_vol)!=0 or np.sum(pred_vol)!=0:
            #     lidc_evaluator.evaluate(np.reshape(masks_vol, [-1]), np.reshape(pred_vol, [-1]))
            print(patient, dscs, ious)
            print(np.sum(masks_vol), np.sum(pred_vol))
            
    total_dscs = np.mean(total_dscs, axis=0)
    total_ious = np.mean(total_ious, axis=0)
    # total_aggregation = lidc_evaluator.get_aggregation(np.mean)
    # mean_iou, mean_dsc = total_aggregation['IoU'], total_aggregation['DSC']
    print(f'IoU: {total_ious} DSC: {total_dscs}')
    print(f'mean IoU: {np.mean(total_ious)} mean DSC: {np.mean(total_dscs)}')


def lidc_volume_generator(cfg):
    # predictor = DefaultPredictor(cfg)
    case_list = dataset_utils.get_files(cfg.FULL_DATA_PATH, recursive=False, get_dirs=True)
    # case_list = case_list[-11:-1]
    case_list = case_list[:10]
    for case_dir in case_list:
        pid = os.path.split(case_dir)[1]
        scan_list = dataset_utils.get_files(os.path.join(case_dir, rf'Image\lung\vol\npy'), 'npy')
        for scan_idx, scan_path in enumerate(scan_list):
            vol = np.load(scan_path)
            mask_vol = np.load(os.path.join(case_dir, rf'Mask\vol\npy', os.path.split(scan_path)[1]))
            infos = {'pid': pid, 'scan_idx': scan_idx}
            yield vol, mask_vol, infos


def asus_nodule_volume_generator(cfg):
    case_list = dataset_utils.get_files(cfg.FULL_DATA_PATH, recursive=False, get_dirs=True)
    case_list = case_list[:10]
    for case_dir in case_list:
        raw_and_mask = dataset_utils.get_files(case_dir, recursive=False, get_dirs=True)
        assert len(raw_and_mask) == 2
        for _dir in raw_and_mask:
            if 'raw' in _dir:
                vol_path = dataset_utils.get_files(_dir, 'mhd', recursive=False)[0]
                vol, _, _ = dataset_utils.load_itk(vol_path)
                vol = np.swapaxes(np.swapaxes(vol, 0, 1), 1, 2)
                for img_idx in range(vol.shape[2]):
                    img = vol[...,img_idx]
                    img = segment_lung(img)
                    if np.max(img)==np.min(img):
                        img = np.zeros_like(img)
                    else:
                        img = np.uint8(255*((img-np.min(img))/(np.max(img)-np.min(img))))
                    vol[...,img_idx] = img
            if 'mask' in _dir:
                vol_mask_path = dataset_utils.get_files(_dir, 'mhd', recursive=False)[0]
                mask_vol, _, _ = dataset_utils.load_itk(vol_mask_path)
                mask_vol = np.swapaxes(np.swapaxes(mask_vol, 0, 1), 1, 2)
        pid = os.path.split(case_dir)[1]
        infos = {'pid': pid, 'scan_idx': 0}
        mask_vol[mask_vol==1] = 2
        yield vol, mask_vol, infos


def luna16_to_lidc(path, key):
    df = pd.read_csv(path)
    index = df.index
    aa = df['Study UID']
    condition = df['Study UID'] == key
    indices = index[condition]
    return indices['Subject ID']

def luna16_volume_generator(cfg):
    subset_list = dataset_utils.get_files(cfg.FULL_DATA_PATH, 'subset', recursive=False, get_dirs=True)
    subset_list = subset_list[:1]
    
    for subset_dir in subset_list:
        case_list = dataset_utils.get_files(subset_dir, 'mhd', recursive=False)
        case_list = case_list[:10]
        for case_dir in case_list:
            # TODO: below same with asus-nodules
            luna16_name = os.path.split(case_dir)[1][:-4]
            lidc_name = luna16_to_lidc(os.path.join(cfg.FULL_DATA_PATH, rf'LIDC-IDRI_MetaData.csv'), key=luna16_name)
            raw_and_mask = dataset_utils.get_files(case_dir, recursive=False, get_dirs=True)
            assert len(raw_and_mask) == 2
            for _dir in raw_and_mask:
                if 'raw' in _dir:
                    vol_path = dataset_utils.get_files(_dir, 'mhd', recursive=False)[0]
                    vol, _, _ = dataset_utils.load_itk(vol_path)
                    vol = np.swapaxes(np.swapaxes(vol, 0, 1), 1, 2)
                    for img_idx in range(vol.shape[2]):
                        img = vol[...,img_idx]
                        img = segment_lung(img)
                        if np.max(img)==np.min(img):
                            img = np.zeros_like(img)
                        else:
                            img = np.uint8(255*((img-np.min(img))/(np.max(img)-np.min(img))))
                        vol[...,img_idx] = img
                if 'mask' in _dir:
                    vol_mask_path = dataset_utils.get_files(_dir, 'mhd', recursive=False)[0]
                    mask_vol, _, _ = dataset_utils.load_itk(vol_mask_path)
                    mask_vol = np.swapaxes(np.swapaxes(mask_vol, 0, 1), 1, 2)
            pid = os.path.split(case_dir)[1]
            infos = {'pid': pid, 'scan_idx': 0}
            mask_vol[mask_vol==1] = 2
            yield vol, mask_vol, infos


# def volume_eval2(cfg, vol_generator):
#     start_time = time.time()
#     predictor = DefaultPredictor(cfg)
#     seg_total_time = 0
#     for vol_idx, (vol, mask_vol, infos) in enumerate(vol_generator(cfg)):
#         pid, scan_idx = infos['pid'], infos['scan_idx']
#         pred_vol = np.zeros_like(mask_vol)
#         for img_idx in range(vol.shape[2]):
#             if img_idx%10 == 0:
#                 print(f'Patient {pid} Scan {scan_idx} slice {img_idx}')
#             seg_start_time = time.time()
#             img = segment_lung(vol[...,img_idx])
#             seg_end_time = time.time()
#             seg_total_time += (seg_end_time-seg_start_time)
#             img[img==-0] = 0
#             img = np.uint8(255*((img-np.min(img))/(1e-7+np.max(img)-np.min(img))))


def volume_eval2(cfg, vol_generator):
    # start_time = time.time()
    predictor = DefaultPredictor(cfg)
    # seg_total_time = 0
    for vol_idx, (vol, mask_vol, infos) in enumerate(vol_generator(cfg)):
        pid, scan_idx = infos['pid'], infos['scan_idx']
        pred_vol = np.zeros_like(mask_vol)
        for img_idx in range(vol.shape[2]):
            if img_idx%10 == 0:
                print(f'Patient {pid} Scan {scan_idx} slice {img_idx}')
            # seg_start_time = time.time()
            img = vol[...,img_idx]
            # img = segment_lung(img)
            # seg_end_time = time.time()
            # seg_total_time += (seg_end_time-seg_start_time)
            # img[img==-0] = 0
            # img = np.uint8(255*((img-np.min(img))/(1e-7+np.max(img)-np.min(img))))
            img = np.uint8(np.tile(img[...,np.newaxis], (1,1,3)))
            outputs = predictor(img) 
            pred = outputs["instances"]._fields['pred_masks'].cpu().detach().numpy() 
            pred_classes = outputs["instances"]._fields['pred_classes'].cpu().detach().numpy() 
            pred_classes = np.reshape(pred_classes, (pred_classes.shape[0], 1, 1))
            pred_classes += 1
            pred = np.sum(pred*pred_classes, axis=0)
            pred = np.int32(pred)
            pred_vol[...,img_idx] = pred

            # Save image for result comparing
            if cfg.SAVE_COMPARE:
                mask = mask_vol[...,img_idx]
                pred = pred_vol[...,img_idx]
                save_name = f'{pid}-{scan_idx}-{img_idx}'
                save_path = os.path.join(cfg.SAVE_PATH, pid)
                save_mask(img, mask, pred, cfg.MODEL.ROI_HEADS.NUM_CLASSES+1, save_path=save_path, save_name=save_name)

        cm = confusion_matrix(np.reshape(mask_vol, [-1]), np.reshape(pred_vol, [-1]), labels=np.arange(0, cfg.MODEL.ROI_HEADS.NUM_CLASSES+1))
        mean_dsc, dscs = metrics2.mean_dsc(cm)
        mean_iou, ious = metrics2.mean_iou(cm)
        print(f'---Patient {pid}  Scan {scan_idx} IoU: {ious} DSC: {dscs}')        
        if vol_idx == 0:
            total_dscs = dscs[np.newaxis]
            total_ious = ious[np.newaxis]
        else:
            total_dscs = np.append(total_dscs, dscs[np.newaxis], axis=0)
            total_ious = np.append(total_ious, ious[np.newaxis], axis=0)

    total_dscs = np.mean(total_dscs, axis=0)
    total_ious = np.mean(total_ious, axis=0)
    print(f'IoU: {total_ious} DSC: {total_dscs}')
    print(f'mean IoU: {np.mean(total_ious):.04f} mean DSC: {np.mean(total_dscs):.04f}')
    # end_time = time.time()

    # print(f'Segment_lung spending total time {seg_total_time:.02f} sec Total {end_time-start_time:.02f} sec ratio {seg_total_time/(end_time-start_time)*100:.02f} %')


# def volume_eval(cfg):
#     predictor = DefaultPredictor(cfg)
#     total_dscs, total_ious = None, None
#     case_list = dataset_utils.get_files(cfg.FULL_DATA_PATH, recursive=False, get_dirs=True)
#     case_list = case_list[-4:-1]
#     # case_list = case_list[:3]

#     for pid_dir in case_list:
#         pid = os.path.split(pid_dir)[1]
#         scan_list = dataset_utils.get_files(os.path.join(pid_dir, rf'Image\lung\vol\npy'), 'npy')
#         for scan_idx, scan_path in enumerate(scan_list):
#             vol = np.load(scan_path)
#             mask_vol = np.load(os.path.join(pid_dir, rf'Mask\vol\npy', os.path.split(scan_path)[1]))
#             pred_vol = np.zeros_like(mask_vol)
#             for img_idx in range(vol.shape[2]):
#                 if img_idx%10 == 0:
#                     print(f'Patient {pid} Scan {scan_idx} slice {img_idx}')
#                 img = vol[...,img_idx]
#                 img = np.uint8(np.tile(img[...,np.newaxis], (1,1,3)))
#                 mask = mask_vol[...,img_idx]

#                 outputs = predictor(img) 
#                 pred = outputs["instances"]._fields['pred_masks'].cpu().detach().numpy() 
#                 pred_classes = outputs["instances"]._fields['pred_classes'].cpu().detach().numpy() 
#                 pred_classes = np.reshape(pred_classes, (pred_classes.shape[0], 1, 1))
#                 pred_classes += 1
#                 pred = np.sum(pred*pred_classes, axis=0)
#                 pred = np.int32(pred)

#                 pred_vol[...,img_idx] = pred
                
#                 # Save image for result comparing
#                 if cfg.SAVE_COMPARE:
#                     save_name = f'{pid}-{scan_idx}-{img_idx}'
#                     save_path = os.path.join(cfg.SAVE_PATH, pid)
#                     save_mask(img, mask, pred, cfg.MODEL.ROI_HEADS.NUM_CLASSES+1, save_path=save_path, save_name=save_name)

#             cm = confusion_matrix(np.reshape(mask_vol, [-1]), np.reshape(pred_vol, [-1]), labels=np.arange(0, cfg.MODEL.ROI_HEADS.NUM_CLASSES+1))
#             mean_dsc, dscs = metrics2.mean_dsc(cm)
#             mean_iou, ious = metrics2.mean_iou(cm)
#             print(f'---Patient {pid}  Scan {scan_idx} IoU: {ious} DSC: {dscs}')        
#             if total_dscs is None:
#                 total_dscs = dscs[np.newaxis]
#                 total_ious = ious[np.newaxis]
#             else:
#                 total_dscs = np.append(total_dscs, dscs[np.newaxis], axis=0)
#                 total_ious = np.append(total_ious, ious[np.newaxis], axis=0)

#     total_dscs = np.mean(total_dscs, axis=0)
#     total_ious = np.mean(total_ious, axis=0)
#     print(f'IoU: {total_ious} DSC: {total_dscs}')
#     print(f'mean IoU: {np.mean(total_ious):.04f} mean DSC: {np.mean(total_dscs):.04f}')



def save_mask(img, mask, pred, num_class, save_path, save_name='img'):
    if np.sum(mask) > 0:
        if np.sum(pred) > 0:
            sub_dir = 'tp'
        else:
            sub_dir = 'fn'
    else:
        if np.sum(pred) > 0:
            sub_dir = 'fp'
        else:
            sub_dir = 'tn'
        
    sub_save_path = os.path.join(save_path, sub_dir)
    if not os.path.isdir(sub_save_path):
        os.makedirs(sub_save_path)

    fig1, _ = compare_result(img, mask, pred, alpha=0.2, vmin=0, vmax=num_class-1)
    fig1.savefig(os.path.join(sub_save_path, f'{save_name}.png'))
    plt.close(fig1)

    fig2, _ = compare_result_enlarge(img, mask, pred, alpha=0.2, vmin=0, vmax=num_class-1)
    if fig2 is not None:
        fig2.savefig(os.path.join(sub_save_path, f'{save_name}-en.png'))
        plt.close(fig2)


def compare_result(image, label, pred, **imshow_params):
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(image)
    ax[0].imshow(label, **imshow_params)
    ax[1].imshow(image)
    ax[1].imshow(pred, **imshow_params)
    ax[0].set_title('Label')
    ax[1].set_title('Prediction')
    return fig, ax


def compare_result_enlarge(image, label, pred, **imshow_params):
    crop_range = 30
    if np.sum(label) > 0:
        item = label
    else:
        if np.sum(pred) > 0:
            item = pred
        else:
            item = None
    
    fig, ax = None, None
    if item is not None:
        h, w, c = image.shape
        ys, xs = np.where(item)
        x1, x2 = max(0, min(xs)-crop_range), min(max(xs)+crop_range, min(h,w))
        y1, y2 = max(0, min(ys)-crop_range), min(max(ys)+crop_range, min(h,w))
        bbox_size = np.max([np.abs(x1-x2), np.abs(y1-y2)])

        image = cv2.resize(image[y1:y1+bbox_size, x1:x1+bbox_size], (w, h), interpolation=cv2.INTER_AREA)
        label = cv2.resize(label[y1:y1+bbox_size, x1:x1+bbox_size], (w, h), interpolation=cv2.INTER_NEAREST)
        pred = cv2.resize(pred[y1:y1+bbox_size, x1:x1+bbox_size], (w, h), interpolation=cv2.INTER_NEAREST)

        fig, ax = compare_result(image, label, pred, **imshow_params)
    return fig, ax


if __name__ == '__main__':
    check_point_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_003'
    check_point_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_010'
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 0
    # cfg.MODEL.WEIGHTS = os.path.join(check_point_path, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.WEIGHTS = os.path.join(check_point_path, "model_0009999.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.OUTPUT_DIR = check_point_path
    cfg.DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI'
    # cfg.INPUT.MIN_SIZE_TEST = 0
    # cfg.INPUT.MAX_SIZE_TEST = 480

    run = os.path.split(check_point_path)[1]
    weight = os.path.split(cfg.MODEL.WEIGHTS)[1].split('.')[0]
    cfg.SAVE_PATH = rf'C:\Users\test\Desktop\Leon\Weekly\1217\maskrcnn-{run}-{weight}-samples'
    # TODO: dataset path in configuration
    # cfg.FULL_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-all-slices'
    # cfg.FULL_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant'
    cfg.FULL_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16'
    cfg.SAVE_COMPARE = False
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    # eval(cfg)
    # save_mask(cfg)
    case_list = dataset_utils.get_files(cfg.DATA_PATH, keys=[], return_fullpath=False, sort=True, recursive=False, get_dirs=True)
    # case_list = case_list[807:]
    # case_list = case_list[810:813]
    # case_list = case_list[815:818]
    # volume_eval3(cfg, case_list)


    # volume_eval2(cfg, vol_generator=asus_nodule_volume_generator)
    volume_eval2(cfg, vol_generator=luna16_volume_generator)
    # volume_eval2(cfg, vol_generator=lidc_volume_generator)