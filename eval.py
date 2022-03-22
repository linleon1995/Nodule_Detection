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
from data.dataloader import SimpleNoduleDataset
from data.data_utils import get_pids_from_coco
from model import build_model
logging.basicConfig(level=logging.INFO)

import site_path
from modules.utils import configuration

from Liwei.LUNA16_test import util
from Liwei.FTP1m_test import test
import cc3d

# TODO: modify the name of lung_mask_filtering and reduce_false_p

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


def detectron2_eval(cfg):
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


def eval(cfg, volume_generator):
    save_path = os.path.join(cfg.SAVE_PATH, cfg.DATASET_NAME, cfg.DATA_SPLIT)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    save_vis_condition = lambda x: True if cfg.SAVE_ALL_COMPARES else True if x < cfg.MAX_SAVE_IMAGE_CASES else False
    total_pid = []
    cls_eval = []
    target_studys, pred_studys = [], []
    lung_mask_path = os.path.join(cfg.DATA_PATH, 'Lung_Mask_show')

    # predictor = BatchPredictor(cfg)
    # TODO:
    predictor = build_model(model_name=cfg.MODEL_NAME, slice_shift=3, n_class=2, pretrained=True, checkpoint_path=cfg.MODEL.WEIGHTS, model_key='net')
    # predictor = build_model(model_name=cfg.MODEL_NAME, slice_shift=3, n_class=2, pretrained=True, checkpoint_path=cfg.MODEL.WEIGHTS, model_key='model_state_dict')

    vol_metric = volumetric_data_eval(model_name=cfg.MODEL_NAME, save_path=save_path, dataset_name=cfg.DATASET_NAME, match_threshold=cfg.MATCHING_THRESHOLD)
    # metadata_recorder = DataFrameTool()
    post_processer = VolumePostProcessor(cfg.connectivity, cfg.area_threshold)
    # nodule_visualizer = Visualizer()
    
    fp_reduce_condition = (cfg.remove_1_slice or cfg.remove_unusual_nodule_by_lung_size or cfg.lung_mask_filtering)
    if fp_reduce_condition:
        fp_reducer = FalsePositiveReducer(_1SR=cfg.remove_1_slice, 
                                          RUNLS=cfg.remove_unusual_nodule_by_lung_size, 
                                          LMF=cfg.lung_mask_filtering, 
                                          slice_threshold=cfg.pred_slice_threshold,
                                          lung_size_threshold=cfg.lung_size_threshold)

    if cfg.nodule_cls:
        crop_range = {'index': cfg.crop_range[0], 'row': cfg.crop_range[1], 'column': cfg.crop_range[2]}
        nodule_classifier = NoduleClassifier(crop_range, cfg.FP_reducer_checkpoint, prob_threshold=cfg.NODULE_CLS_PROB)


    for vol_idx, (raw_vol, vol, mask_vol, infos) in enumerate(volume_generator):
        # TODO: use decorator to write a breaking condition
        if cfg.MAX_TEST_CASES is not None:
            if vol_idx >= cfg.MAX_TEST_CASES:
                break
            
        pid, scan_idx = infos['pid'], infos['scan_idx']
        total_pid.append(pid)
        mask_vol = np.int32(mask_vol)
        
        # Model Inference
        print(f'\n Volume {vol_idx} Patient {pid} Scan {scan_idx}')
        
        dataset = SimpleNoduleDataset(vol, slice_shift=3)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)
        pred_vol = pytorch_model_inference(vol, predictor, dataloader)
        # pred_vol = d2_model_inference(vol, batch_size=cfg.TEST_BATCH_SIZE, predictor=predictor)
        # pred_vol = model_inference(cfg.MODEL_NAME, vol, batch_size=cfg.TEST_BATCH_SIZE, predictor=predictor)

        # Data post-processing
        # TODO: the target volume should reduce small area but 1 pixel remain in 1m0037 
        pred_vol_category = post_processer(pred_vol)
        # target_vol_individual = post_processer.connect_components(mask_vol, connectivity=cfg.connectivity)
        target_vol_individual = post_processer(mask_vol)

        # False positive reducing
        if fp_reduce_condition:
            pred_vol_category = fp_reducer(pred_vol_category, raw_vol, lung_mask_path, pid)

        # Nodule classification
        if cfg.nodule_cls:
            pred_vol_category, pred_nodule_info = nodule_classifier.nodule_classify(vol, pred_vol_category, mask_vol)
        else:
            pred_nodule_info = None
        # Evaluation
        target_study = LungNoduleStudy(pid, target_vol_individual, raw_volume=raw_vol)
        pred_study = LungNoduleStudy(pid, pred_vol_category, raw_volume=raw_vol)
        vol_metric.calculate(target_study, pred_study)

        # print('test')
        # TODO: single function
        # # Visualize
        if save_vis_condition(vol_idx):
            origin_save_path = os.path.join(save_path, 'images', pid, 'origin')
            enlarge_save_path = os.path.join(save_path, 'images', pid, 'enlarge')
            _3d_save_path = os.path.join(save_path, 'images', pid, '3d')
            for path in [origin_save_path, enlarge_save_path, _3d_save_path]:
                if not os.path.isdir(path):
                    os.makedirs(path)

            # # for j, (img, mask, pred) in enumerate(zip(vol, mask_vol, pred_vol)):
            # #     save_mask(img, mask, pred, num_class=2, save_path=image_save_path, save_name=f'{pid}-{j:03d}')
            
            # # npy_save_path = os.path.join(save_path, 'npy')
            # # if not os.path.isdir(npy_save_path):
            # #     os.makedirs(npy_save_path)
            # # np.save(os.path.join(npy_save_path, f'{pid}.npy'), np.uint8(pred_vol))

            vis_vol, vis_indices, vis_crops = visualize(vol, pred_vol_category, mask_vol, pred_nodule_info)
            for vis_idx in vis_indices:
                # plt.savefig(vis_vol[vis_idx])
                cv2.imwrite(os.path.join(origin_save_path, f'vis-{pid}-{vis_idx}.png'), vis_vol[vis_idx])
                for crop_idx, vis_crop in enumerate(vis_crops[vis_idx]):
                    cv2.imwrite(os.path.join(enlarge_save_path, f'vis-{pid}-{vis_idx}-crop{crop_idx:03d}.png'), vis_crop)

            temp = np.where(mask_vol+pred_vol>0, 1, 0)
            zs_c, ys_c, xs_c = np.where(temp)
            crop_range = {'z': (np.min(zs_c), np.max(zs_c)), 'y': (np.min(ys_c), np.max(ys_c)), 'x': (np.min(xs_c), np.max(xs_c))}
            if crop_range['z'][1]-crop_range['z'][0] > 2 and \
               crop_range['y'][1]-crop_range['y'][0] > 2 and \
               crop_range['x'][1]-crop_range['x'][0] > 2:
                save_mask_in_3d(target_vol_individual, 
                                save_path1=os.path.join(_3d_save_path, f'{pid}-raw-mask.png'),
                                save_path2=os.path.join(_3d_save_path, f'{pid}-preprocess-mask.png'), 
                                crop_range=crop_range)
                save_mask_in_3d(pred_vol_category,
                                save_path1=os.path.join(_3d_save_path, f'{pid}-raw-pred.png'),
                                save_path2=os.path.join(_3d_save_path, f'{pid}-preprocess-pred.png'),
                                crop_range=crop_range)
        target_studys.append(target_study)
        pred_studys.append(pred_study)
        #     nodule_visualizer()

    _ = vol_metric.evaluation(show_evaluation=True)
    return target_studys, pred_studys

    
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
    

def select_model(cfg):
    checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_001'
    checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_003'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_004'
    checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_005'
    checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_006'
    checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_007'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_010'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_016'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_017'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_018'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_019'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_020'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_021'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_022'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_023'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_024'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_026'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_027'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_028'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_032'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_033'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_034'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_035'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_036'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_037'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_040'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_041'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_044'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_045'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_046'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_048'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_049'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_051'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_053'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_052'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_055'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_056'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_057'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_058'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_059'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_060'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_061'
    cfg.OUTPUT_DIR = checkpoint_path

    cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0005999.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0005999.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_final.pth")  # path to the model we just trained


    cfg.OUTPUT_DIR = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\run_002'
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "ckpt_best.pth")  # path to the model we just trained
    # cfg.OUTPUT_DIR = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\liwei'
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best.pt")  # path to the model we just trained
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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set a custom testing threshold
    cfg.MATCHING_THRESHOLD = 0.1
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.INPUT.MIN_SIZE_TEST = 1120
    cfg.DATA_SPLIT = 'test'
    cfg.NODULE_CLS_PROB = 0.75
    cfg.connectivity = 26
    cfg.area_threshold = 8
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4,  8,  16,  32,  64]]
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.2]]
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8,  16,  32,  64, 128]]
    # cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 20
    # cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 20
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    # False Positive reduction
    cfg.nodule_cls = False
    cfg.crop_range = [48, 48, 48]
    cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\run_011\ckpt_best.pth'
    cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\run_001\ckpt_best.pth'
    cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\run_004\ckpt_best.pth'
    cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\run_010\ckpt_best.pth'
    cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\run_016\ckpt_best.pth'

    cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\run_019\ckpt_best.pth'
    # cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\run_020\ckpt_best.pth'
    # cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\run_021\ckpt_best.pth'
    # cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\run_022\ckpt_best.pth'
    # cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\run_023\ckpt_best.pth'
    # cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\run_023\ckpt_best.pth'
    # cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\run_027\ckpt_best.pth'
    cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\run_028\ckpt_best.pth'
    # cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\run_033\ckpt_best.pth'

    cfg.lung_mask_filtering = False
    cfg.remove_1_slice = False
    cfg.remove_unusual_nodule_by_lung_size = False
    cfg.remove_unusual_nodule_by_ratio = False
    cfg.lung_size_threshold = 0.4
    cfg.pred_slice_threshold = 1
    
    run = os.path.split(cfg.OUTPUT_DIR)[1]
    weight = os.path.split(cfg.MODEL.WEIGHTS)[1].split('.')[0]
    # cfg.SAVE_PATH = rf'C:\Users\test\Desktop\Leon\Weekly\1227'
    dir_name = ['maskrcnn', f'{run}', f'{weight}', f'{cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}']
    FPR_model_code = os.path.split(os.path.split(cfg.FP_reducer_checkpoint)[0])[1]
    dir_name.insert(0, '1SR') if cfg.remove_1_slice else dir_name
    dir_name.insert(0, 'RUNR') if cfg.remove_unusual_nodule_by_ratio else dir_name
    dir_name.insert(0, f'RUNLS_TH{cfg.lung_size_threshold}') if cfg.remove_unusual_nodule_by_lung_size else dir_name
    dir_name.insert(0, 'LMF') if cfg.lung_mask_filtering else dir_name
    dir_name.insert(0, f'NC#{FPR_model_code}') if cfg.nodule_cls else dir_name
    dir_name.insert(0, str(cfg.INPUT.MIN_SIZE_TEST))
    cfg.SAVE_PATH = os.path.join(cfg.OUTPUT_DIR, '-'.join(dir_name))
    cfg.MAX_SAVE_IMAGE_CASES = 3
    cfg.MAX_TEST_CASES = None
    cfg.ONLY_NODULES = True
    cfg.SAVE_ALL_COMPARES = False
    cfg.TEST_BATCH_SIZE = 1

    return cfg


# def nodule_test():
#     cfg = common_config()

#     cfg.RAW_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\benign_merge'
#     cfg.DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\benign\raw_merge'

#     # cfg.RAW_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant_merge'
#     # cfg.DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\malignant\raw_merge'

#     cfg = add_dataset_name(cfg)
#     cfg.DATA_SPLIT = 'test'

#     cfg.SUBSET_INDICES = None
#     if cfg.DATA_SPLIT == 'train':
#         # cfg.CASE_INDICES = list(range(25))
#         cfg.CASE_INDICES = list(range(17))
#     elif cfg.DATA_SPLIT == 'valid':
#         # cfg.CASE_INDICES = list(range(25, 27))
#         cfg.CASE_INDICES = list(range(17, 19))
#     elif cfg.DATA_SPLIT == 'test':
#         # cfg.CASE_INDICES = list(range(27, 35))
#         cfg.CASE_INDICES = list(range(19, 25))
#     else:
#         cfg.CASE_INDICES = None

#     train_generator = asus_nodule_volume_generator(cfg.RAW_DATA_PATH, 
#                                                    subset_indices=cfg.SUBSET_INDICES, 
#                                                    case_indices=list(range(17)),
#                                                 #    case_indices=list(range(34)),
#                                                    only_nodule_slices=cfg.ONLY_NODULES)
#     test_generator = asus_nodule_volume_generator(cfg.RAW_DATA_PATH, 
#                                                   subset_indices=cfg.SUBSET_INDICES, 
#                                                   case_indices=list(range(19, 25)),
#                                                 #   case_indices=list(range(36, 44)),
#                                                   only_nodule_slices=cfg.ONLY_NODULES)

#     from data import data_analysis

#     train_volumes = []
#     for vol_idx, (raw_vol, vol, mask_vol, infos) in enumerate(train_generator):
#         train_volumes.append(mask_vol)

#     test_volumes = []
#     for vol_idx, (raw_vol, vol, mask_vol, infos) in enumerate(test_generator):
#         test_volumes.append(mask_vol)

#     data_analysis.multi_nodule_distribution(train_volumes, test_volumes)
#     # data_analysis.multi_nodule_distribution(test_volumes[:3], test_volumes[3:])


def cross_valid_eval():
    test_cfg = configuration.load_config(f'config_file/test.yml', dict_as_member=True)

    dataset_names = test_cfg.DATA.NAMES

    model_name = test_cfg.MODEL.NAME

    cfg = common_config()

    model_weight = cfg.MODEL.WEIGHTS
    save_path = cfg.SAVE_PATH
    cfg.MODEL_NAME = model_name
    # TODO: move to config
    assign_fold = 4

    if assign_fold is not None:
        assert assign_fold < test_cfg.CV_FOLD, 'Assign fold out of range'
        fold_indices = [assign_fold]
    else:
        fold_indices = list(range(test_cfg.CV_FOLD))

    benign_target_scatter_vis = ScatterVisualizer()
    benign_pred_scatter_vis = ScatterVisualizer()
    malignant_target_scatter_vis = ScatterVisualizer()
    malignant_pred_scatter_vis = ScatterVisualizer()
    benign_rcorder = DataFrameTool(
        column_name=['study_id', 'type', 'nodule_id', 'avg_hu', 'size', 'index', 'row', 'column', 'IoU', 'DSC', 'Best_Slice_IoU', 'tp', 'fp', 'fn'])
    malignant_rcorder = DataFrameTool(
        column_name=['study_id', 'type', 'nodule_id', 'avg_hu', 'size', 'index', 'row', 'column', 'IoU', 'DSC', 'Best_Slice_IoU', 'tp', 'fp', 'fn'])


    for fold in fold_indices:
        for dataset_name in dataset_names:
            coco_path = os.path.join(test_cfg.PATH.DATA_ROOT[dataset_name], 'coco', test_cfg.TASK_NAME, f'cv-{test_cfg.CV_FOLD}', str(fold))
            # coco_path = os.path.join(rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\Annotations\ASUS_Nodule', dataset_name)
            
            case_pids = get_pids_from_coco(os.path.join(coco_path, f'annotations_{test_cfg.DATA.SPLIT}.json'))
            cfg.RAW_DATA_PATH = os.path.join(test_cfg.PATH.DATA_ROOT[dataset_name], 'merge')
            cfg.DATA_PATH = os.path.join(test_cfg.PATH.DATA_ROOT[dataset_name], 'image')
            cfg.DATASET_NAME = dataset_name
            cfg.FOLD = fold
            
            # cfg.MODEL.WEIGHTS = os.path.join(os.path.split(model_weight)[0], str(cfg.FOLD), os.path.split(model_weight)[1])
            # cfg.MODEL.WEIGHTS = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\Liwei\FTP1m_test\model\FCN_all_best.pt'

            # cfg.SAVE_PATH = os.path.join(os.path.split(save_path)[0], str(cfg.FOLD), os.path.split(save_path)[1])
            # cfg.SAVE_PATH = os.path.join(os.path.split(cfg.MODEL.WEIGHTS)[0])

            volume_generator = asus_nodule_volume_generator(cfg.RAW_DATA_PATH, 
                                                            case_pids=case_pids)
            target_studys, pred_studys = eval(cfg, volume_generator=volume_generator)

            for target_study, pred_study in zip(target_studys, pred_studys):
                if dataset_name == 'ASUS-Benign':
                    benign_target_scatter_vis.record(target_study)
                    benign_pred_scatter_vis.record(pred_study)
                    
                elif dataset_name == 'ASUS-Malignant':
                    malignant_target_scatter_vis.record(target_study)
                    malignant_pred_scatter_vis.record(pred_study)

                for target_nodule_id in target_study.nodule_instances:
                    target_nodule = target_study.nodule_instances[target_nodule_id]
                    if dataset_name == 'ASUS-Benign':
                        benign_data = [target_study.study_id, 'target', target_nodule.id, target_nodule.hu, target_nodule.nodule_size, 
                                       target_nodule.nodule_center['index'], target_nodule.nodule_center['row'], target_nodule.nodule_center['column'], 
                                       target_nodule.nodule_score['IoU'], target_nodule.nodule_score['DSC'], 0,
                                       target_study.get_score('NoduleTP'), target_study.get_score('NoduleFP'), target_study.get_score('NoduleFN')]
                        benign_rcorder.write_row(benign_data)
                    elif dataset_name == 'ASUS-Malignant':
                        malignant_data = [target_study.study_id, 'target', target_nodule.id, target_nodule.hu, target_nodule.nodule_size, 
                                          target_nodule.nodule_center['index'], target_nodule.nodule_center['row'], target_nodule.nodule_center['column'], 
                                          target_nodule.nodule_score['IoU'], target_nodule.nodule_score['DSC'], 0,
                                          target_study.get_score('NoduleTP'), target_study.get_score('NoduleFP'), target_study.get_score('NoduleFN')]
                        malignant_rcorder.write_row(malignant_data)

                for pred_nodule_id in pred_study.nodule_instances:
                    pred_nodule = pred_study.nodule_instances[pred_nodule_id]
                    if dataset_name == 'ASUS-Benign':
                        benign_data = [pred_study.study_id, 'pred', pred_nodule.id, pred_nodule.hu, pred_nodule.nodule_size, 
                                       pred_nodule.nodule_center['index'], pred_nodule.nodule_center['row'], pred_nodule.nodule_center['column'], 
                                       pred_nodule.nodule_score['IoU'], pred_nodule.nodule_score['DSC'], 0, 
                                       pred_study.get_score('NoduleTP'), pred_study.get_score('NoduleFP'), pred_study.get_score('NoduleFN')]
                        benign_rcorder.write_row(benign_data)
                    elif dataset_name == 'ASUS-Malignant':
                        malignant_data = [pred_study.study_id, 'pred', pred_nodule.id, pred_nodule.hu, pred_nodule.nodule_size, 
                                          pred_nodule.nodule_center['index'], pred_nodule.nodule_center['row'], pred_nodule.nodule_center['column'], 
                                          pred_nodule.nodule_score['IoU'], pred_nodule.nodule_score['DSC'], 0, 
                                          pred_study.get_score('NoduleTP'), pred_study.get_score('NoduleFP'), pred_study.get_score('NoduleFN')]
                        malignant_rcorder.write_row(malignant_data)


    benign_target_scatter_vis.show_scatter(save_path=os.path.join(cfg.OUTPUT_DIR, 'benign_target_nodule.png'), 
                                           title='Benign target nodule', xlabel='size (pixels)', ylabel='meanHU')
    benign_pred_scatter_vis.show_scatter(save_path=os.path.join(cfg.OUTPUT_DIR, 'benign_pred_nodule.png'), 
                                         title='Benign predict nodule', xlabel='size (pixels)', ylabel='meanHU')
    malignant_target_scatter_vis.show_scatter(save_path=os.path.join(cfg.OUTPUT_DIR, 'malignant_target_nodule.png'), 
                                              title='Malignant target nodule', xlabel='size (pixels)', ylabel='meanHU')
    malignant_pred_scatter_vis.show_scatter(save_path=os.path.join(cfg.OUTPUT_DIR, 'malignant_pred_nodule.png'), 
                                            title='Malignant predict nodule', xlabel='size (pixels)', ylabel='meanHU')
    
    # TODO: file rename
    benign_rcorder.save_data_frame(save_path=os.path.join(cfg.SAVE_PATH, 'benign.csv'))
    malignant_rcorder.save_data_frame(save_path=os.path.join(cfg.SAVE_PATH, 'malignant.csv'))


def main():
    cross_valid_eval()
    
if __name__ == '__main__':
    main()


    # nodule_test()
    # asus_benign_eval()
    # asus_malignant_eval()
    # luna16_eval()

    # liwei_asus_malignant_eval()
    
    
    