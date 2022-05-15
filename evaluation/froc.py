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
from torch.utils.data import Dataset, DataLoader
from zmq import device

from data import volume_generator
import config

# from data.luna16_crop_preprocess import LUNA16_CropRange_Builder
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from skimage import measure
# from numpy.lib.npyio import save
# from pylidc.utils import consensus
# from pathlib import Path
# from statistics import median_high
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils.utils import cv2_imshow, calculate_malignancy, segment_lung, mask_preprocess
from utils.utils import raw_preprocess, compare_result, compare_result_enlarge, time_record
# from convert_to_coco_structure import lidc_to_datacatlog_valid
import logging
from sklearn.metrics import confusion_matrix
# import time
# import pylidc as pl
import pandas as pd
# from tqdm import tqdm
# import json
from utils.volume_eval import volumetric_data_eval
from utils.utils import Nodule_data_recording, DataFrameTool, SubmissionDataFrame, irc2xyz, get_nodule_center
from utils.vis import save_mask, visualize, save_mask_in_3d, plot_scatter, ScatterVisualizer
# import liwei_eval
from evaluationScript import noduleCADEvaluationLUNA16
from reduce_false_positive import NoduleClassifier
from data.data_postprocess import VolumePostProcessor
from data.volume_generator import luna16_volume_generator, asus_nodule_volume_generator, build_pred_generator
from inference import model_inference, d2_model_inference, pytorch_model_inference
from lung_mask_filtering import get_lung_mask, remove_unusual_nodule_by_ratio, remove_unusual_nodule_by_lung_size, _1_slice_removal, FalsePositiveReducer
from data.data_structure import LungNoduleStudy
from data.dataloader import GeneralDataset, SimpleNoduleDataset, CropNoduleDataset
from data.data_utils import get_pids_from_coco
from utils.evaluator import Pytorch2dSegEvaluator, Pytorch3dSegEvaluator, D2SegEvaluator
from eval import BatchPredictor
from model import build_model
logging.basicConfig(level=logging.INFO)

import site_path
from modules.utils import configuration

from Liwei.LUNA16_test import util
from Liwei.FTP1m_test import test
import cc3d



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