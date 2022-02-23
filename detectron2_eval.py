from detectron2.evaluation import COCOEvaluator, inference_on_dataset, SemSegEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
import torch
import os
import cv2
import random
import numpy as np
import matplotlib as mpl
import argparse

from data.luna16_data_preprocess import LUNA16_CropRange_Builder
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

from utils.utils import cv2_imshow, calculate_malignancy, segment_lung, mask_preprocess
from utils.utils import raw_preprocess, compare_result, compare_result_enlarge, time_record
# from convert_to_coco_structure import lidc_to_datacatlog_valid
import logging
from sklearn.metrics import confusion_matrix
import time
import pylidc as pl
import pandas as pd
from tqdm import tqdm
from utils.volume_generator import luna16_volume_generator, asus_nodule_volume_generator, build_pred_generator
from utils.volume_eval import volumetric_data_eval
from utils.utils import Nodule_data_recording, SubmissionDataFrame, irc2xyz, get_nodule_center
from utils.vis import save_mask
import liwei_eval
from evaluationScript import noduleCADEvaluationLUNA16
from reduce_false_positive import False_Positive_Reducer
from lung_mask_filtering import get_lung_mask
logging.basicConfig(level=logging.INFO)

import site_path
from modules.data import dataset_utils
# from modules.utils import metrics
from modules.utils import metrics2
from modules.utils import evaluator

from Liwei.LUNA16_test import util
from Liwei.FTP1m_test import test


def volume_outputs_to_pred_volume(volume_outputs):
    volume_shape = [len(volume_outputs)] + list(volume_outputs[0].image_size)
    pred_vol = np.zeros(volume_shape, np.float32)
    for img_idx, instance in enumerate(volume_outputs):
        pred_masks = instance._fields['pred_masks'].cpu().detach().numpy() 
        pred_mask = np.sum(pred_masks, axis=0)
        pred_mask = mask_preprocess(pred_mask)
        pred_vol[img_idx] = pred_mask
    return pred_vol

def volume_outputs_to_pred_scores(volume_outputs):
    all_slices_scores = []
    for img_idx, instance in enumerate(volume_outputs):
        scores = instance._fields['scores'].cpu().detach().numpy() 
        all_slices_scores.append(scores)
    return all_slices_scores

def convert_pred_format(volume_outputs):
    pred_vol = volume_outputs_to_pred_volume(volume_outputs)
    pred_scores = volume_outputs_to_pred_scores(volume_outputs)

    # TODO: the affect need to be confirmed
    # Remove single slice prediction
    # score_filter = []
    # for img_idx, scores in enumerate(pred_scores):
    #     if scores.size > 0:
    #         last, next = max(0, img_idx-1), min(len(pred_scores)-1, img_idx+1)
    #         if pred_scores[last].size == 0 and pred_scores[next].size == 0:
    #             pred_scores[img_idx] = np.array([], np.float32)
    #             score_filter.append(0)
    #         else:
    #             score_filter.append(1)
    #     else:
    #         score_filter.append(0)
    # score_filter = np.array(score_filter, np.float32)
    # score_filter = np.reshape(score_filter, [score_filter.size, 1, 1])
    # pred_vol *= score_filter
    return pred_vol, pred_scores


def get_output(pred_vol, pred_scores, origin, spacing, direction): 
    pred_vol, pred_metadata = volumetric_data_eval.volume_preprocess(pred_vol)
    pred_category = np.unique(pred_vol)[1:]
    total_nodule_infos = []
    for label in pred_category:
        pred_nodule = np.where(pred_vol==label, 1, 0)
        pred_center_irc = get_nodule_center(pred_nodule)
        pred_center_xyz = irc2xyz(pred_center_irc, origin, spacing, direction)
        # TODO: The nodule prob is actually the mean of all nodule probs of assign slice.
        # This is suboptimal but temporally acceptable solution.
        # Because nodule based prob need to convert 2d bbox to 3d bbox which is hard to implement
        nodule_prob = np.mean(pred_scores[int(pred_center_irc[0])])
        nodule_infos= {'Center_xyz': pred_center_xyz, 'Nodule_prob': nodule_prob}
        total_nodule_infos.append(nodule_infos)
    return total_nodule_infos


class BatchPredictor(DefaultPredictor):
    """Run d2 on a list of images."""
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, images):
        """Run d2 on a list of images.

        Args:
            images (list): BGR images of the expected shape: 720x1280
        """
        transform = []
        for origin_image in images:
            origin_image = origin_image[0]
            image = self.aug.get_transform(origin_image).apply_image(origin_image)
            transform.append({'image': torch.as_tensor(image.astype("float32").transpose(2, 0, 1)), 
                              'height': 512, 'width': 512})
        images = transform
        
        # images = [
        #     {'image': torch.as_tensor(image[0].astype("float32").transpose(2, 0, 1))}
        #     for image in images
        # ]
        with torch.no_grad():
            preds = self.model(images)
        return preds


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


def volume_eval(cfg, volume_generator):
    # TODO: Add trial number to divide different trial (csv will replace)
    # TODO: predictor, volume, generator,... should be input into this function rather define in the function
    # TODO: Select a better Data interface or implement both (JSON, volume loading)
    # TODO: save_image_condition
    save_path = os.path.join(cfg.SAVE_PATH, cfg.DATASET_NAME)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    save_image_condition = lambda x: True if cfg.SAVE_ALL_COMPARES else True if x < cfg.MAX_SAVE_IMAGE_CASES else False
    total_pid = []

    time_recording = time_record()
    time_recording.set_start_time('Total')
    # predictor = liwei_eval.liwei_predictor
    predictor = BatchPredictor(cfg)
    vol_metric = volumetric_data_eval(save_path)
    metadata_recorder = Nodule_data_recording()
    lung_mask_path = os.path.join(cfg.DATA_PATH, 'Lung_Mask_show')

    crop_range = {'index': cfg.crop_range[0], 'row': cfg.crop_range[1], 'column': cfg.crop_range[2]}
    if cfg.reduce_false_positive:
        FP_reducer = False_Positive_Reducer(crop_range, cfg.FP_reducer_checkpoint)

    # submission_recorder = SubmissionDataFrame()
    
    # volume_generator = vol_generator(cfg.RAW_DATA_PATH, subset_indices=cfg.SUBSET_INDICES, case_indices=cfg.CASE_INDICES,
    #                                  only_nodule_slices=cfg.ONLY_NODULES)

    # xx = luna16_volume_generator(cfg.RAW_DATA_PATH, subset_indices=cfg.SUBSET_INDICES, case_indices=cfg.CASE_INDICES)
    # total_pid = xx.pid_list
    for vol_idx, (raw_vol, vol, mask_vol, infos) in enumerate(volume_generator):
        pid, scan_idx = infos['pid'], infos['scan_idx']
        total_pid.append(pid)
        mask_vol = np.int32(mask_vol)
        pred_vol = np.zeros_like(mask_vol)
        
        image_save_path = os.path.join(save_path, 'images', pid)
        # TODO: use decorator to write a breaking condition
        if cfg.MAX_TEST_CASES is not None:
            if vol_idx >= cfg.MAX_TEST_CASES:
                break

        for batch_start_index in range(0, vol.shape[0], cfg.TEST_BATCH_SIZE):
            if batch_start_index == 0:
                print(f'\n Volume {vol_idx} Patient {pid} Scan {scan_idx} Slice {batch_start_index}')
            start, end = batch_start_index, min(vol.shape[0], batch_start_index+cfg.TEST_BATCH_SIZE)
            img = vol[start:end]
            time_recording.set_start_time('Inference')
            img_list = np.split(img, img.shape[0], axis=0)
            outputs = predictor(img_list) 
            time_recording.set_end_time('Inference')

            for j, output in enumerate(outputs):
                pred = output["instances"]._fields['pred_masks'].cpu().detach().numpy() 
                pred = np.sum(pred, axis=0)
                # TODO: better way to calculate pred score of slice
                # pred_score = np.mean(output["instances"]._fields['scores'].cpu().detach().numpy() , axis=0)
                pred = mask_preprocess(pred)
                img_idx = batch_start_index + j
                pred_vol[img_idx] = pred

        time_recording.set_start_time('Nodule Evaluation')

        if cfg.lung_mask_filtering:
            lung_mask_case_path = os.path.join(lung_mask_path, pid)

            if not os.path.isdir(lung_mask_case_path):
                os.makedirs(lung_mask_case_path)
                lung_mask_vol = get_lung_mask(pred_vol, raw_vol[...,0])
                for lung_mask_idx, lung_mask in enumerate(lung_mask_vol):
                    cv2.imwrite(os.path.join(lung_mask_case_path, f'{pid}-{lung_mask_idx:03d}.png'), 255*lung_mask)
            else:
                lung_mask_files = dataset_utils.get_files(lung_mask_case_path, 'png')
                lung_mask_vol = np.zeros_like(pred_vol)
                for lung_mask_idx, lung_mask in enumerate(lung_mask_files): 
                    lung_mask_vol[lung_mask_idx] = cv2.imread(lung_mask)[...,0]
                lung_mask_vol = lung_mask_vol / 255
                lung_mask_vol = np.int32(lung_mask_vol)
            pred_vol *= lung_mask_vol


        if cfg.reduce_false_positive:
            pred_vol = FP_reducer.reduce_false_positive(vol, pred_vol)

        vol_nodule_infos = vol_metric.calculate(mask_vol, pred_vol, infos)
        # TODO: Not clean, is dict order correctly? (Pid has to be the first place)
        # vol_nodule_infos = {'Nodule_pid': pid}.update(vol_nodule_infos)

        # TODO: check behavior: save_image_condition
        if save_image_condition(vol_idx):
            for j, (img, mask, pred) in enumerate(zip(vol, mask_vol, pred_vol)):
                time_recording.set_start_time('Save result in image.')
                save_mask(img, mask, pred, num_class=2, save_path=image_save_path, save_name=f'{pid}-{j:03d}')
                time_recording.set_end_time('Save result in image.')

        for nodule_infos in vol_nodule_infos:
            # if nodule_infos['Nodule_prob']:
                # submission = [pid] + nodule_infos['Center_xyz'].tolist() + [nodule_infos['Nodule_prob']]
                # submission_recorder.write_row(submission)
            nodule_infos.pop('Center_xyz', None)
            nodule_infos.pop('Nodule_prob', None)
        metadata_recorder.write_row(vol_nodule_infos, pid)
        time_recording.set_end_time('Nodule Evaluation')

        # save_sample_submission(vol_nodule_infos)
        # if pid == pid_list[-1]:
        #     break
        # save_mask_in_3d(mask_vol, 
        #                 save_path1=os.path.join(case_save_path, f'{pid}-{img_idx:03d}-raw-mask.png'),
        #                 save_path2=os.path.join(case_save_path, f'{pid}-{img_idx:03d}-preprocess-mask.png'))
        # save_mask_in_3d(pred_vol, 
        #                 save_path1=os.path.join(case_save_path, f'{pid}-{img_idx:03d}-raw-pred.png'),
        #                 save_path2=os.path.join(case_save_path, f'{pid}-{img_idx:03d}-preprocess-pred.png'))
        
    print(cfg.DATASET_NAME, os.path.split(cfg.OUTPUT_DIR)[1], cfg.DATA_SPLIT, os.path.split(cfg.MODEL.WEIGHTS)[1], 
          cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    nodule_tp, nodule_fp, nodule_fn, nodule_precision, nodule_recall = vol_metric.evaluation(show_evaluation=True)
    # submission_recorder.save_data_frame(save_path=os.path.join(cfg.SAVE_PATH, 'FROC', f'{cfg.DATASET_NAME}-submission.csv'))
    df = metadata_recorder.get_data_frame()
    df.to_csv(os.path.join(save_path, f'{cfg.DATA_SPLIT}-{cfg.DATASET_NAME}-nodule_informations.csv'), index=False)
    time_recording.set_end_time('Total')
    time_recording.show_recording_time()


def froc(cfg, volume_generator):
    time_recording = time_record()
    time_recording.set_start_time('Total')

    save_path = os.path.join(cfg.SAVE_PATH, cfg.DATASET_NAME)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    pid_list = []

    predictor = BatchPredictor(cfg)
    vol_metric = volumetric_data_eval(save_path)
    submission_recorder = SubmissionDataFrame()
    idx = 0
    for pid, volume_outputs, nodule_infos in build_pred_generator(volume_generator, predictor, cfg.TEST_BATCH_SIZE):
        if cfg.MAX_TEST_CASES is not None:
            if idx > cfg.MAX_TEST_CASES: 
                break
        idx += 1
        pid_list.append(pid)
        pred_vol, pred_scores = convert_pred_format(volume_outputs)
        vol_nodule_infos = get_output(pred_vol, pred_scores, 
                                      nodule_infos['origin'], nodule_infos['spacing'], nodule_infos['direction'])

        for nodule_infos in vol_nodule_infos:
            if not nodule_infos['Nodule_prob']: 
                nodule_infos['Nodule_prob'] = 0.5
            submission = [pid] + nodule_infos['Center_xyz'].tolist() + [nodule_infos['Nodule_prob']]
            submission_recorder.write_row(submission)

    submission_recorder.save_data_frame(save_path=os.path.join(save_path, 'FROC', f'{cfg.DATA_SPLIT}-{cfg.DATASET_NAME}-submission.csv'))
    time_recording.set_end_time('Total')

    seriesuid = pd.DataFrame(data=pid_list)
    annotation_dir = os.path.join(save_path, 'FROC', 'annotations')
    if not os.path.isdir(annotation_dir):
        os.makedirs(annotation_dir)
    seriesuid.to_csv(os.path.join(save_path, 'FROC', 'annotations', 'seriesuids.csv'), index=False, header=False)

    CalculateFROC(f'{cfg.DATA_SPLIT}-{cfg.DATASET_NAME}-submission', save_path)


# def froc(cfg, volume_generator):
#     time_recording = time_record()
#     time_recording.set_start_time('Total')

#     save_path = os.path.join(cfg.SAVE_PATH, cfg.DATASET_NAME)

#     if not os.path.isdir(save_path):
#         os.makedirs(save_path)
#     pid_list = []

#     predictor = BatchPredictor(cfg)
#     vol_metric = volumetric_data_eval(save_path)
#     submission_recorder = SubmissionDataFrame()
#     for infos, mask_vol, pred_vol in build_pred_generator(volume_generator, predictor, cfg.TEST_BATCH_SIZE):
#         pid_list.append(infos['pid'])
#         vol_nodule_infos = vol_metric.get_submission(mask_vol, pred_vol, infos)
#         # if infos['vol_idx'] > 1: break
#         for nodule_infos in vol_nodule_infos:
#             if not nodule_infos['Nodule_prob']: 
#                 nodule_infos['Nodule_prob'] = 0.5
#             submission = [infos['pid']] + nodule_infos['Center_xyz'].tolist() + [nodule_infos['Nodule_prob']]
#             submission_recorder.write_row(submission)

#     submission_recorder.save_data_frame(save_path=os.path.join(save_path, 'FROC', f'{cfg.DATA_SPLIT}-{cfg.DATASET_NAME}-submission.csv'))
#     time_recording.set_end_time('Total')

#     seriesuid = pd.DataFrame(data=pid_list)
#     annotation_dir = os.path.join(save_path, 'FROC', 'annotations')
#     if not os.path.isdir(annotation_dir):
#         os.makedirs(annotation_dir)
#     seriesuid.to_csv(os.path.join(save_path, 'FROC', 'annotations', 'seriesuids.csv'), index=False, header=False)

#     CalculateFROC(f'{cfg.DATA_SPLIT}-{cfg.DATASET_NAME}-submission', save_path)


def CalculateFROC(submission_filename, save_path):
    annotation_dir = os.path.join(save_path, 'FROC', 'annotations')

    # select calculate cases from seriesuid.csv
    annotations = pd.read_csv('evaluationScript/annotations/annotations.csv')
    annotations_excluded = pd.read_csv('evaluationScript/annotations/annotations_excluded.csv')
    calculate_pid = pd.read_csv(os.path.join(annotation_dir, 'seriesuids.csv'))
    calculate_annotations = annotations.loc[annotations['seriesuid'].isin(calculate_pid.iloc[:, 0])]
    calculate_annotations_excluded = annotations_excluded.loc[annotations_excluded['seriesuid'].isin(calculate_pid.iloc[:, 0])]

    # save annotation and annotation_excluded
    calculate_annotations.to_csv(os.path.join(annotation_dir, 'annotations.csv'), index=False)
    calculate_annotations_excluded.to_csv(os.path.join(annotation_dir, 'annotations_excluded.csv'), index=False)

    noduleCADEvaluationLUNA16.noduleCADEvaluation(os.path.join(annotation_dir, 'annotations.csv'),
                                                  os.path.join(annotation_dir, 'annotations_excluded.csv'),
                                                #   'evaluationScript/annotations/seriesuids2.csv',
                                                  os.path.join(annotation_dir, 'seriesuids.csv'),
                                                  os.path.join(save_path, 'FROC', f'{submission_filename}.csv'),
                                                  os.path.join(save_path, 'FROC'))


# def save_sample_submission(vol_nodule_infos):
    # TODO: class
    # TODO: no index in the first column
    # TODO: coordX, Y, Z --> calculate center in volume_eval and coord transform in here (use nodule DSC as prob)
    

def select_model(cfg):
    checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_003'
    checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_004'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_006'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_010'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_016'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_017'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_018'
    checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_019'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_020'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_021'
    checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_022'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_023'
    checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_024'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_026'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_032'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_033'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_034'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_035'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_036'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_037'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_040'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_041'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_044'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_045'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_046'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_048'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_049'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_051'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_053'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_052'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_055'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_056'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_057'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_058'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_059'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_060'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_061'
    cfg.OUTPUT_DIR = checkpoint_path

    cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0039999.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_final.pth")  # path to the model we just trained
    return cfg

def add_dataset_name(cfg):
    for dataset_name in ['LUNA16', 'ASUS', 'LIDC']:
        if dataset_name in cfg.RAW_DATA_PATH:
            if dataset_name == 'ASUS':
                if 'benign' in cfg.RAW_DATA_PATH:
                    dataset_name = f'{dataset_name}-benign'
                elif 'malignant' in cfg.RAW_DATA_PATH:
                    dataset_name = f'{dataset_name}-malignant'
            break
        dataset_name = None

    assert dataset_name is not None, 'Unknown dataset name.'
    cfg.DATASET_NAME = dataset_name
    return cfg


def common_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg = select_model(cfg)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.INPUT.MIN_SIZE_TEST = 1120
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4,  8,  16,  32,  64]]
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.2]]
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8,  16,  32,  64, 128]]
    # cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 20
    # cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 20
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    # False Positive reduction
    cfg.reduce_false_positive = False
    cfg.crop_range = [48, 48, 48]
    cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\checkpoints\run_011\ckpt_best.pth'
    cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\checkpoints\run_001\ckpt_best.pth'
    cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\checkpoints\run_004\ckpt_best.pth'
    cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\checkpoints\run_010\ckpt_best.pth'
    cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\checkpoints\run_016\ckpt_best.pth'

    cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\checkpoints\run_019\ckpt_best.pth'
    # cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\checkpoints\run_020\ckpt_best.pth'
    # cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\checkpoints\run_021\ckpt_best.pth'
    # cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\checkpoints\run_022\ckpt_best.pth'
    # cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\checkpoints\run_023\ckpt_best.pth'
    # cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\checkpoints\run_023\ckpt_best.pth'
    # cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\checkpoints\run_027\ckpt_best.pth'
    cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\checkpoints\run_028\ckpt_best.pth'

    cfg.lung_mask_filtering = False
    
    run = os.path.split(cfg.OUTPUT_DIR)[1]
    weight = os.path.split(cfg.MODEL.WEIGHTS)[1].split('.')[0]
    # cfg.SAVE_PATH = rf'C:\Users\test\Desktop\Leon\Weekly\1227'
    dir_name = ['maskrcnn', f'{run}', f'{weight}', f'{cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}']
    FPR_model_code = os.path.split(os.path.split(cfg.FP_reducer_checkpoint)[0])[1]
    dir_name.insert(0, f'FPR_{FPR_model_code}') if cfg.reduce_false_positive else dir_name
    dir_name.insert(0, 'LMF') if cfg.lung_mask_filtering else dir_name
    dir_name.insert(0, str(cfg.INPUT.MIN_SIZE_TEST))
    cfg.SAVE_PATH = os.path.join(cfg.OUTPUT_DIR, '-'.join(dir_name))
    cfg.MAX_SAVE_IMAGE_CASES = 1
    cfg.MAX_TEST_CASES = None
    cfg.ONLY_NODULES = True
    cfg.SAVE_ALL_COMPARES = False
    cfg.TEST_BATCH_SIZE = 2

    return cfg


def luna16_eval():
    cfg = common_config()
    cfg.RAW_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data'
    cfg.DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\raw'
    cfg = add_dataset_name(cfg)
    cfg.DATA_SPLIT = 'test'

    if cfg.DATA_SPLIT == 'train':
        cfg.SUBSET_INDICES = list(range(7))
    elif cfg.DATA_SPLIT == 'valid':
        cfg.SUBSET_INDICES = [7]
    elif cfg.DATA_SPLIT == 'test':
        cfg.SUBSET_INDICES = [8, 9]
    else:
        cfg.SUBSET_INDICES = None
    cfg.CASE_INDICES = None

    volume_generator = luna16_volume_generator.Build_DLP_luna16_volume_generator(
        cfg.RAW_DATA_PATH, subset_indices=cfg.SUBSET_INDICES, case_indices=cfg.CASE_INDICES, only_nodule_slices=cfg.ONLY_NODULES)
    volume_eval(cfg, volume_generator=volume_generator)
    # froc(cfg, volume_generator)
    # CalculateFROC(cfg.DATASET_NAME, cfg.SAVE_PATH)
    # calculateFROC(cfg)
    # eval(cfg)
    return cfg


def asus_malignant_eval():
    cfg = common_config()
    cfg.RAW_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant_merge'
    cfg.DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\malignant\raw_merge'
    cfg = add_dataset_name(cfg)
    cfg.DATA_SPLIT = 'test'

    cfg.SUBSET_INDICES = None
    if cfg.DATA_SPLIT == 'train':
        cfg.CASE_INDICES = list(range(34))
        # cfg.CASE_INDICES = list(range(40))
    elif cfg.DATA_SPLIT == 'valid':
        cfg.CASE_INDICES = list(range(34, 36))
        # cfg.CASE_INDICES = list(range(40, 45))
    elif cfg.DATA_SPLIT == 'test':
        cfg.CASE_INDICES = list(range(36, 44))
        # cfg.CASE_INDICES = list(range(45, 57))
    else:
        cfg.CASE_INDICES = None

    volume_generator = asus_nodule_volume_generator(cfg.RAW_DATA_PATH, subset_indices=cfg.SUBSET_INDICES, case_indices=cfg.CASE_INDICES,
                                     only_nodule_slices=cfg.ONLY_NODULES)
    volume_eval(cfg, volume_generator=volume_generator)
    return cfg


def asus_benign_eval():
    cfg = common_config()
    cfg.RAW_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\benign_merge'
    cfg.DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\benign\raw_merge'
    cfg = add_dataset_name(cfg)
    cfg.DATA_SPLIT = 'test'

    cfg.SUBSET_INDICES = None
    if cfg.DATA_SPLIT == 'train':
        # cfg.CASE_INDICES = list(range(25))
        cfg.CASE_INDICES = list(range(17))
    elif cfg.DATA_SPLIT == 'valid':
        # cfg.CASE_INDICES = list(range(25, 27))
        cfg.CASE_INDICES = list(range(17, 18))
    elif cfg.DATA_SPLIT == 'test':
        # cfg.CASE_INDICES = list(range(27, 35))
        cfg.CASE_INDICES = list(range(19, 25))
    else:
        cfg.CASE_INDICES = None

    volume_generator = asus_nodule_volume_generator(cfg.RAW_DATA_PATH, subset_indices=cfg.SUBSET_INDICES, case_indices=cfg.CASE_INDICES,
                                     only_nodule_slices=cfg.ONLY_NODULES)
    volume_eval(cfg, volume_generator=volume_generator)
    return cfg


if __name__ == '__main__':
    # asus_benign_eval()
    # asus_malignant_eval()
    luna16_eval()

    # liwei_asus_malignant_eval()
    
    
    