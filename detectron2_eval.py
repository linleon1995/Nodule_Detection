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

from utils import cv2_imshow, calculate_malignancy, segment_lung, mask_preprocess, raw_preprocess, compare_result, compare_result_enlarge
from convert_to_coco_structure import lidc_to_datacatlog_valid
import logging
from sklearn.metrics import confusion_matrix
import time
import pylidc as pl
import pandas as pd
from tqdm import tqdm
from volume_generator import luna16_volume_generator, lidc_volume_generator, asus_nodule_volume_generator
from volume_eval import volumetric_data_eval
logging.basicConfig(level=logging.INFO)

import site_path
from modules.data import dataset_utils
# from modules.utils import metrics
from modules.utils import metrics2
from modules.utils import evaluator

from LUNA16_test import util


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


def eval2(cfg):
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    predictor = DefaultPredictor(cfg)

    register_coco_instances("my_dataset_valid", {}, rf"Annotations\LUNA16\annotations_valid.json", cfg.DATA_PATH)
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


def volume_eval5(cfg, vol_generator):
    predictor = DefaultPredictor(cfg)
    # metric = util.metrics(n_class=2)
    vol_metric = volumetric_data_eval()
    # volume_generator = vol_generator(cfg.FULL_DATA_PATH, only_nodule_slices=cfg.ONLY_NODULES)
    volume_generator = vol_generator(cfg.FULL_DATA_PATH, subset_indices=cfg.SUBSET_INDICES, case_indices=cfg.CASE_INDICES,
                                     only_nodule_slices=cfg.ONLY_NODULES)
    for vol_idx, (vol, mask_vol, infos) in enumerate(volume_generator):
        pid, scan_idx = infos['pid'], infos['scan_idx']
        mask_vol = np.int32(mask_vol)
        pred_vol = np.zeros_like(mask_vol)
        # if vol_idx > 4: break
        # print(pid)
        # if pid != '1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405':
        #     continue
        for img_idx in range(vol.shape[0]):
            if img_idx%20 == 0:
                print(f'Volume {vol_idx} Patient {pid} Scan {scan_idx} Slice {img_idx}')
            img = vol[img_idx]
            outputs = predictor(img) 
            pred = outputs["instances"]._fields['pred_masks'].cpu().detach().numpy() 
            # if pred.shape[0] > 0:
            #     print(3)
            pred = np.sum(pred, axis=0)
            pred = mask_preprocess(pred)
            pred_vol[img_idx] = pred

            if cfg.SAVE_COMPARE:
                if vol_idx > 4:
                    continue
                save_path = os.path.join(cfg.SAVE_PATH, pid)
                save_mask(img, mask_vol[img_idx], pred, num_class=2, save_path=save_path, save_name=f'{pid}-{img_idx:03d}.png')
        # metric.calculate(pred_vol, mask_vol, area_th=10)
        vol_metric.calculate(mask_vol, pred_vol)
        
    nodule_tp, nodule_fp, nodule_fn, nodule_precision, nodule_recall = vol_metric.evaluation(show_evaluation=True)
    print(30*'=')
    # class_acc, class_iou, class_f1, mIOU, pixel_Precision, pixel_Recall, Total_dice = metric.evaluation(True)


def nodules_eval(pid, pred_vol):
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid)
    num_scan_in_one_patient = scans.count()
    print(f'{pid} has {num_scan_in_one_patient} scan')
    scan_list = scans.all()
    mask_threshold = 8

    # TODO:
    scan_list = scan_list[:1]
    for scan_idx, scan in enumerate(scan_list):
        if scan is None:
            print(scan)
        nodules_annotation = scan.cluster_annotations()
        
        print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid, pred_vol.shape, len(nodules_annotation)))

        # Patients with nodules
        masks_vol = np.zeros_like(pred_vol)
        total_dsc, total_iou = np.array([], np.float32), np.array([], np.float32)
        for nodule_idx, nodule in enumerate(nodules_annotation):
        # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
        # This current for loop iterates over total number of nodules in a single patient
            nodule_mask_vol, cbbox, masks = consensus(nodule, clevel=0.5, pad=512)
            # malignancy, cancer_label = calculate_malignancy(nodule)
            # one_mask_vol = np.zeros_like(vol)
            nodule_pred_vol = pred_vol[cbbox]
            total_cm = 0
            for slice_idx in range(nodule_mask_vol.shape[2]):
                if np.sum(nodule_mask_vol[:,:,slice_idx]) <= mask_threshold:
                    continue
                # print(np.sum(nodule_mask_vol[...,slice_idx]), np.sum(nodule_pred_vol[...,slice_idx]))
                # print(np.max(nodule_mask_vol[...,slice_idx]), np.max(nodule_pred_vol[...,slice_idx]))
                total_cm += confusion_matrix(np.reshape(nodule_mask_vol[...,slice_idx], [-1]), np.reshape(nodule_pred_vol[...,slice_idx], [-1]))
            mean_dsc, dscs = metrics2.mean_dsc(total_cm)
            mean_iou, ious = metrics2.mean_iou(total_cm)
            total_dsc = np.append(total_dsc, mean_dsc)
            total_iou = np.append(total_iou, mean_iou)
        # print('IoU', total_iou)
        # print('DSC', total_dsc)
    return total_iou, total_dsc
                

    

def save_mask(img, mask, pred, num_class, save_path, save_name='img'):
    # if np.sum(mask) > 0:
    #     if np.sum(pred) > 0:
    #         sub_dir = 'tp'
    #     else:
    #         sub_dir = 'fn'
    # else:
    #     if np.sum(pred) > 0:
    #         sub_dir = 'fp'
    #     else:
    #         sub_dir = 'tn'
        
    # sub_save_path = os.path.join(save_path, sub_dir)
    sub_save_path = save_path
    if not os.path.isdir(sub_save_path):
        os.makedirs(sub_save_path)

    fig1, _ = compare_result(img, mask, pred, show_mask_size=True, alpha=0.2, vmin=0, vmax=num_class-1)
    fig1.savefig(os.path.join(sub_save_path, f'{save_name}.png'))
    # fig1.tight_layout()
    plt.close(fig1)

    fig2, _ = compare_result_enlarge(img, mask, pred, show_mask_size=False, alpha=0.2, vmin=0, vmax=num_class-1)
    if fig2 is not None:
        fig2.savefig(os.path.join(sub_save_path, f'{save_name}-en.png'))
        # fig2.tight_layout()
        plt.close(fig2)



if __name__ == '__main__':
    check_point_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_003'
    check_point_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_010'
    check_point_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_019'
    check_point_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_023'
    check_point_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_026'
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 0
    # cfg.MODEL.WEIGHTS = os.path.join(check_point_path, "model_final.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(check_point_path, "model_0003999.pth")  # path to the model we just trained
    cfg.MODEL.WEIGHTS = os.path.join(check_point_path, "model_0007999.pth")  # path to the model we just trained
    cfg.MODEL.WEIGHTS = os.path.join(check_point_path, "model_0015999.pth")  # path to the model we just trained
    cfg.MODEL.WEIGHTS = os.path.join(check_point_path, "model_0019999.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(check_point_path, "model_0069999.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.OUTPUT_DIR = check_point_path
    # cfg.DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI'
    # cfg.INPUT.MIN_SIZE_TEST = 0
    # cfg.INPUT.MAX_SIZE_TEST = 480

    run = os.path.split(check_point_path)[1]
    weight = os.path.split(cfg.MODEL.WEIGHTS)[1].split('.')[0]
    cfg.SAVE_PATH = rf'C:\Users\test\Desktop\Leon\Weekly\1227\maskrcnn-{run}-{weight}-{cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}-samples'
    # TODO: dataset path in configuration
    cfg.FULL_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-all-slices'
    cfg.FULL_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data'
    # cfg.FULL_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant'

    cfg.DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\raw'
    cfg.SAVE_COMPARE = False
    # cfg.CASE_INDICES = list(range(10))
    cfg.SUBSET_INDICES = [8, 9]
    # cfg.SUBSET_INDICES = [2]
    # cfg.SUBSET_INDICES = [0, 1]
    # cfg.SUBSET_INDICES = list(range(8))
    # cfg.CASE_INDICES = list(range(10, 20))
    cfg.CASE_INDICES = None
    # cfg.CASE_INDICES = list(range(810, 820))
    cfg.ONLY_NODULES = True
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    # eval2(cfg)
    # save_mask(cfg)
    # case_list = dataset_utils.get_files(cfg.DATA_PATH, keys=[], return_fullpath=False, sort=True, recursive=False, get_dirs=True)
    # case_list = case_list[807:]
    # case_list = case_list[810:813]
    # case_list = case_list[815:818]
    # volume_eval3(cfg, case_list)


    # volume_eval2(cfg, vol_generator=asus_nodule_volume_generator)
    # volume_eval2(cfg, vol_generator=luna16_volume_generator)
    # volume_eval2(cfg, vol_generator=lidc_volume_generator)


    # volume_eval4(cfg, vol_generator=lidc_volume_generator)
    # volume_eval4(cfg, vol_generator=asus_nodule_volume_generator)

    volume_eval5(cfg, vol_generator=luna16_volume_generator)
    # volume_eval5(cfg, vol_generator=asus_nodule_volume_generator)