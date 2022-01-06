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
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from numpy.lib.npyio import save
from pylidc.utils import consensus
from pathlib import Path
from statistics import median_high
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import cv2_imshow, calculate_malignancy, segment_lung, mask_preprocess
from utils import raw_preprocess, compare_result, compare_result_enlarge, time_record
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

    # register_coco_instances("my_dataset_valid", {}, "annotations_valid.json", rf"C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-Preprocessing-png\Image")
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


def volume_eval(cfg, vol_generator):
    time_recording = time_record()
    time_recording.set_start_time('Total')
    predictor = DefaultPredictor(cfg)
    vol_metric = volumetric_data_eval()
    
    volume_generator = vol_generator(cfg.FULL_DATA_PATH, subset_indices=cfg.SUBSET_INDICES, case_indices=cfg.CASE_INDICES,
                                     only_nodule_slices=cfg.ONLY_NODULES)

    for vol_idx, (vol, mask_vol, infos) in enumerate(volume_generator):
        pid, scan_idx = infos['pid'], infos['scan_idx']
        mask_vol = np.int32(mask_vol)
        pred_vol = np.zeros_like(mask_vol)
        case_save_path = os.path.join(cfg.SAVE_PATH, pid)
        # if vol_idx > 10:
        #     break
        # pid_list =[
        #             '1.3.6.1.4.1.14519.5.2.1.6279.6001.229860476925100292554329427970',
        #             '1.3.6.1.4.1.14519.5.2.1.6279.6001.204287915902811325371247860532',
        #             '1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860',
        #            '1.3.6.1.4.1.14519.5.2.1.6279.6001.387954549120924524005910602207']
        # if pid not in pid_list:
        #     continue
        
        time_recording.set_start_time('2D Model Inference'if not cfg.SAVE_COMPARE else '2D Model Inference and Save result in image.')
        for img_idx in range(vol.shape[0]):
            if img_idx%50 == 0:
                print(f'Volume {vol_idx} Patient {pid} Scan {scan_idx} Slice {img_idx}')
            img = vol[img_idx]
            outputs = predictor(img) 
            pred = outputs["instances"]._fields['pred_masks'].cpu().detach().numpy() 
            pred = np.sum(pred, axis=0)
            pred = mask_preprocess(pred)
            pred_vol[img_idx] = pred
            if cfg.SAVE_COMPARE:
                save_mask(img, mask_vol[img_idx], pred, num_class=2, save_path=case_save_path, save_name=f'{pid}-{img_idx:03d}.png')
        time_recording.set_end_time('2D Model Inference'if not cfg.SAVE_COMPARE else '2D Model Inference and Save result in image.')

        time_recording.set_start_time('Nodule Evaluation')
        vol_nodule_infos = vol_metric.calculate(mask_vol, pred_vol)
        if vol_idx == 0:
            nodule_idx = 0
            vol_info_attritube = list(vol_nodule_infos[0].keys())
            vol_info_attritube.insert(0, 'Series uid')
            vol_info_attritube.extend(['IoU>0.1', 'IoU>0.3', 'IoU>0.5', 'IoU>0.7', 'IoU>0.9'])
            df = pd.DataFrame(columns=vol_info_attritube)
        for nodule_info in vol_nodule_infos:
            vol_info_value = list(nodule_info.values())
            vol_info_value.insert(0, pid)
            vol_info_value.extend([np.int32(nodule_info['Nodule IoU']>0.1), 
                                   np.int32(nodule_info['Nodule IoU']>0.3), 
                                   np.int32(nodule_info['Nodule IoU']>0.5), 
                                   np.int32(nodule_info['Nodule IoU']>0.7), 
                                   np.int32(nodule_info['Nodule IoU']>0.9)])
            df.loc[nodule_idx] = vol_info_value
            nodule_idx += 1
            print(nodule_info)
        time_recording.set_end_time('Nodule Evaluation')
        # if pid == pid_list[-1]:
        #     break
        # save_mask_in_3d(mask_vol, 
        #                 save_path1=os.path.join(case_save_path, f'{pid}-{img_idx:03d}-raw-mask.png'),
        #                 save_path2=os.path.join(case_save_path, f'{pid}-{img_idx:03d}-preprocess-mask.png'))
        # save_mask_in_3d(pred_vol, 
        #                 save_path1=os.path.join(case_save_path, f'{pid}-{img_idx:03d}-raw-pred.png'),
        #                 save_path2=os.path.join(case_save_path, f'{pid}-{img_idx:03d}-preprocess-pred.png'))
        
    nodule_tp, nodule_fp, nodule_fn, nodule_precision, nodule_recall = vol_metric.evaluation(show_evaluation=True)
    df.to_csv(os.path.join(cfg.SAVE_PATH, 'nodule_informations.csv'))
    time_recording.set_end_time('Total')
    time_recording.show_recording_time()
    

def save_mask_in_3d_interface(vol_generator, save_path1, save_path2):
    volume_generator = vol_generator(cfg.FULL_DATA_PATH, subset_indices=cfg.SUBSET_INDICES, case_indices=cfg.CASE_INDICES,
                                     only_nodule_slices=cfg.ONLY_NODULES)
    for vol_idx, (vol, mask_vol, infos) in enumerate(volume_generator):
        pid, scan_idx = infos['pid'], infos['scan_idx']
        mask_vol = np.int32(mask_vol)
        if vol_idx > 9:
            if np.sum(mask_vol==0) == mask_vol.size:
                print('No mask')
                continue

            save_mask_in_3d(mask_vol, save_path1, save_path2)


def save_mask_in_3d(volume, save_path1, save_path2):
    if np.sum(volume==0) == volume.size:
        print('No mask')
    else:
        plot_volume_in_mesh(volume, 0, save_path1)
        volume = volumetric_data_eval.volume_preprocess(volume, connectivity=26, area_threshold=30)
        print(np.unique(volume))
        volume_list = [np.int32(volume==label) for label in np.unique(volume)[1:]]
        plot_volume_in_mesh(volume_list, 0, save_path2)


def show_mask_in_2d(cfg, vol_generator):
    volume_generator = vol_generator(cfg.FULL_DATA_PATH, subset_indices=cfg.SUBSET_INDICES, case_indices=cfg.CASE_INDICES,
                                     only_nodule_slices=cfg.ONLY_NODULES)
    fig, ax = plt.subplots(1, 1, figsize=(4,4))
    for vol_idx, (vol, mask_vol, infos) in enumerate(volume_generator):
        pid, scan_idx = infos['pid'], infos['scan_idx']
        mask_vol = np.int32(mask_vol)
        if vol_idx in [0, 2, 3, 7, 10 ,11]:
            if np.sum(mask_vol==0) == mask_vol.size:
                print('No mask')
                continue
            mask_vol2 = volumetric_data_eval.volume_preprocess(mask_vol, connectivity=26, area_threshold=30)
            def plot_func(volume, name):
                zs, ys, xs = np.where(volume)
                # min_nonzero_slice, max_nonzero_slice = np.min(zs), np.max(zs)
                zs = np.unique(zs)
                for slice_idx in zs:
                    ax.imshow(volume[slice_idx], vmin=0, vmax=5)
                    fig.savefig(os.path.join(cfg.SAVE_PATH, '2d_mask', f'{name}-{vol_idx:03d}-{slice_idx:03d}.png'))

            plot_func(mask_vol, 'raw')
            plot_func(mask_vol2, 'preprocess')


def plot_volume_in_mesh(volume_geroup, threshold=-300, save_path=None): 
    if not isinstance(volume_geroup, list):
        volume_geroup = [volume_geroup]

    # TODO: fix limited colors
    # from itertools import combinations
 
    # # Get all combinations of [1, 2, 3]
    # # and length 2
    # comb = combinations([1, 2, 3], 2)
    
    # # Print the obtained combinations
    # for i in list(comb):
    #     print (i)
    colors = [[0.5, 0.5, 1], [0.5, 1, 0.5], [1, 0.5, 0.5], [0.5, 1, 1], [1, 1, 0.5], [1, 0.5, 1],
              [0.1, 0.7, 1], [0.7, 1, 0.1], [1, 0.7, 0.1], [0.1, 0.7, 0.7], [0.7, 0.7, 0.1], [0.7, 0.1, 0.7]]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for vol_idx, vol in enumerate(volume_geroup):
        p = vol.transpose(2,1,0)
        verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
        mesh = Poly3DCollection(verts[faces], alpha=0.3)
        face_color = colors[vol_idx]
        # face_color = np.random.rand(3)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_3d(image, threshold=-300): 
    p = image.transpose(2,1,0)
    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.3)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()


def save_mask(img, mask, pred, num_class, save_path, save_name='img', mask_or_pred_exist=True):
    if mask_or_pred_exist:
        condition = (np.sum(mask)>0 or np.sum(pred)>0)
    else:
        condition = True
        
    if condition:
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
    check_point_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_032'
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
    cfg.SAVE_COMPARE = True
    # cfg.CASE_INDICES = list(range(10))
    cfg.SUBSET_INDICES = [8, 9]
    # cfg.INPUT.MIN_SIZE_TEST = 512
    # cfg.SUBSET_INDICES = [2]
    # cfg.SUBSET_INDICES = [0, 1]
    # cfg.SUBSET_INDICES = list(range(8))
    # cfg.CASE_INDICES = list(range(10, 20))
    cfg.CASE_INDICES = None
    # cfg.CASE_INDICES = list(range(810, 820))
    cfg.ONLY_NODULES = True
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    # eval(cfg)
    volume_eval(cfg, vol_generator=luna16_volume_generator)
    # volume_eval(cfg, vol_generator=asus_nodule_volume_generator)
    # show_mask_in_3d(vol_generator=luna16_volume_generator)
    # show_mask_in_2d(cfg, vol_generator=luna16_volume_generator)