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
# from convert_to_coco_structure import lidc_to_datacatlog_valid
import logging
from sklearn.metrics import confusion_matrix
import time
import pylidc as pl
import pandas as pd
from tqdm import tqdm
from volume_generator import luna16_volume_generator, lidc_volume_generator, asus_nodule_volume_generator, build_pred_generator
from volume_eval import volumetric_data_eval
from utils import Nodule_data_recording, SubmissionDataFrame
from vis import save_mask
import liwei_eval
from evaluationScript import noduleCADEvaluationLUNA16
logging.basicConfig(level=logging.INFO)

import site_path
from modules.data import dataset_utils
# from modules.utils import metrics
from modules.utils import metrics2
from modules.utils import evaluator

from LUNA16_test import util


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



# def volume_eval2(cfg, volume_generator):
#     if not os.path.isdir(cfg.SAVE_PATH):
#         os.makedirs(cfg.SAVE_PATH)
#     vol_metric = volumetric_data_eval()
#     for mask_vol, pred_vol, vol_nodule_infos in build_pred_generator(cfg, volume_generator):
#         vol_nodule_infos = vol_metric.calculate(mask_vol, pred_vol, infos)
    
#         for nodule_infos in vol_nodule_infos:
#             if nodule_infos['Nodule_prob']:
#                 submission = [pid] + nodule_infos['Center_xyz'].tolist() + [nodule_infos['Nodule_prob']]
#                 submission_recorder.write_row(submission)
#             nodule_infos.pop('Center_xyz', None)
#             nodule_infos.pop('Nodule_prob', None)
#         metadata_recorder.write_row(vol_nodule_infos, pid)
#         time_recording.set_end_time('Nodule Evaluation')

#         # save_sample_submission(vol_nodule_infos)
#         # if pid == pid_list[-1]:
#         #     break
#         # save_mask_in_3d(mask_vol, 
#         #                 save_path1=os.path.join(case_save_path, f'{pid}-{img_idx:03d}-raw-mask.png'),
#         #                 save_path2=os.path.join(case_save_path, f'{pid}-{img_idx:03d}-preprocess-mask.png'))
#         # save_mask_in_3d(pred_vol, 
#         #                 save_path1=os.path.join(case_save_path, f'{pid}-{img_idx:03d}-raw-pred.png'),
#         #                 save_path2=os.path.join(case_save_path, f'{pid}-{img_idx:03d}-preprocess-pred.png'))
        
#     print(cfg.DATASET_NAME, os.path.split(cfg.checkpoint_path)[1], cfg.DATA_SPLIT, os.path.split(cfg.MODEL.WEIGHTS)[1], 
#           cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
#     nodule_tp, nodule_fp, nodule_fn, nodule_precision, nodule_recall = vol_metric.evaluation(show_evaluation=True)
#     submission_recorder.save_data_frame(save_path=os.path.join(cfg.SAVE_PATH, 'FROC', f'{cfg.DATASET_NAME}-submission.csv'))
#     df = metadata_recorder.get_data_frame()
#     df.to_csv(os.path.join(cfg.SAVE_PATH, f'{cfg.DATASET_NAME}-nodule_informations.csv'), index=False)
#     time_recording.set_end_time('Total')
#     time_recording.show_recording_time()


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
    vol_metric = volumetric_data_eval()
    metadata_recorder = Nodule_data_recording()
        
    # submission_recorder = SubmissionDataFrame()
    
    # volume_generator = vol_generator(cfg.RAW_DATA_PATH, subset_indices=cfg.SUBSET_INDICES, case_indices=cfg.CASE_INDICES,
    #                                  only_nodule_slices=cfg.ONLY_NODULES)

    # xx = luna16_volume_generator(cfg.RAW_DATA_PATH, subset_indices=cfg.SUBSET_INDICES, case_indices=cfg.CASE_INDICES)
    # total_pid = xx.pid_list
    for vol_idx, (vol, mask_vol, infos) in enumerate(volume_generator):
        pid, scan_idx = infos['pid'], infos['scan_idx']
        total_pid.append(pid)
        mask_vol = np.int32(mask_vol)
        pred_vol = np.zeros_like(mask_vol)
        
        image_save_path = os.path.join(save_path, 'images', pid)
        # TODO: use decorator to write a breaking condition
        if cfg.MAX_TEST_CASES is not None:
            if vol_idx >= cfg.MAX_TEST_CASES:
                break

        for img_idx in range(0, vol.shape[0], cfg.TEST_BATCH_SIZE):
            if img_idx == 0:
                print(f'\n Volume {vol_idx} Patient {pid} Scan {scan_idx} Slice {img_idx}')
            start, end = img_idx, min(vol.shape[0], img_idx+cfg.TEST_BATCH_SIZE)
            img = vol[start:end]
            time_recording.set_start_time('Inference')
            img = np.split(img, img.shape[0], axis=0)
            outputs = predictor(img) 
            time_recording.set_end_time('Inference')


            for j, output in enumerate(outputs):
                pred = output["instances"]._fields['pred_masks'].cpu().detach().numpy() 
                pred = np.sum(pred, axis=0)
                # TODO: better way to calculate pred score of slice
                # pred_score = np.mean(output["instances"]._fields['scores'].cpu().detach().numpy() , axis=0)
                pred = mask_preprocess(pred)
                pred_vol[img_idx+j] = pred

                time_recording.set_start_time('Save result in image.')
                # TODO: check behavior: save_image_condition
                if save_image_condition(vol_idx):
                    save_mask(img[j][0], mask_vol[img_idx+j], pred, num_class=2, save_path=image_save_path, save_name=f'{pid}-{img_idx+j:03d}.png')
                time_recording.set_end_time('Save result in image.')

        time_recording.set_start_time('Nodule Evaluation')
        vol_nodule_infos = vol_metric.calculate(mask_vol, pred_vol, infos)
        # TODO: Not clean, is dict order correctly? (Pid has to be the first place)
        # vol_nodule_infos = {'Nodule_pid': pid}.update(vol_nodule_infos)

        
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
        
    print(cfg.DATASET_NAME, os.path.split(cfg.checkpoint_path)[1], cfg.DATA_SPLIT, os.path.split(cfg.MODEL.WEIGHTS)[1], 
          cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    nodule_tp, nodule_fp, nodule_fn, nodule_precision, nodule_recall = vol_metric.evaluation(show_evaluation=True)
    # submission_recorder.save_data_frame(save_path=os.path.join(cfg.SAVE_PATH, 'FROC', f'{cfg.DATASET_NAME}-submission.csv'))
    df = metadata_recorder.get_data_frame()
    df.to_csv(os.path.join(save_path, f'{cfg.DATA_SPLIT}-{cfg.DATASET_NAME}-nodule_informations.csv'), index=False)
    time_recording.set_end_time('Total')
    time_recording.show_recording_time()


def froc(cfg, volume_generator):
    # time_recording = time_record()
    # time_recording.set_start_time('Total')
    save_path = os.path.join(cfg.SAVE_PATH, cfg.DATASET_NAME)
    # if not os.path.isdir(save_path):
    #     os.makedirs(save_path)
    # pid_list = []

    # predictor = BatchPredictor(cfg)
    # vol_metric = volumetric_data_eval()
    # submission_recorder = SubmissionDataFrame()
    # for infos, mask_vol, pred_vol in build_pred_generator(volume_generator, predictor, cfg.TEST_BATCH_SIZE):
    #     pid_list.append(infos['pid'])
    #     vol_nodule_infos = vol_metric.get_submission(mask_vol, pred_vol, infos)

    #     for nodule_infos in vol_nodule_infos:
    #         if not nodule_infos['Nodule_prob']: 
    #             nodule_infos['Nodule_prob'] = 0.5
    #         submission = [infos['pid']] + nodule_infos['Center_xyz'].tolist() + [nodule_infos['Nodule_prob']]
    #         submission_recorder.write_row(submission)

    # submission_recorder.save_data_frame(save_path=os.path.join(save_path, 'FROC', f'{cfg.DATA_SPLIT}-{cfg.DATASET_NAME}-submission.csv'))
    # time_recording.set_end_time('Total')

    # seriesuid = pd.DataFrame(data=pid_list)
    # seriesuid.to_csv(os.path.join(save_path, 'FROC', 'annotations', 'seriesuids.csv'), index=False, header=False)
    CalculateFROC(f'{cfg.DATA_SPLIT}-{cfg.DATASET_NAME}-submission', save_path)
    # time_recording.show_recording_time()


def CalculateFROC(submission_filename, save_path):
    annotation_dir = os.path.join(save_path, 'FROC', 'annotations')
    if not os.path.isdir(annotation_dir):
        os.makedirs(annotation_dir)

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
    checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_010'
    checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_019'
    checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_023'
    checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_026'
    checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_032'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_034'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_035'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_036'
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_037'
    checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_033'
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
    cfg.checkpoint_path = checkpoint_path
    
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0000399.pth")  # path to the model we just trained
    cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0000799.pth")  # path to the model we just trained
    cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0000999.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0001199.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0001599.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0001999.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0003999.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0005999.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0007999.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0009999.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0011999.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0015999.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0019999.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0023999.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0027999.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0039999.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0059999.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0069999.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path, "model_0079999.pth")  # path to the model we just trained
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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.OUTPUT_DIR = cfg.checkpoint_path
    cfg.INPUT.MIN_SIZE_TEST = 800
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4,  8,  16,  32,  64]]
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.2]]
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8,  16,  32,  64, 128]]
    # cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 32
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    run = os.path.split(cfg.checkpoint_path)[1]
    weight = os.path.split(cfg.MODEL.WEIGHTS)[1].split('.')[0]
    cfg.SAVE_PATH = rf'C:\Users\test\Desktop\Leon\Weekly\1227\maskrcnn-{run}-{weight}-{cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}-samples'
    cfg.MAX_SAVE_IMAGE_CASES = 100
    cfg.MAX_TEST_CASES = None
    cfg.ONLY_NODULES = True
    cfg.SAVE_ALL_COMPARES = False
    cfg.TEST_BATCH_SIZE = 1
    return cfg


def luna16_eval():
    cfg = common_config()
    cfg.RAW_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data'
    cfg.DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\raw'
    cfg = add_dataset_name(cfg)
    cfg.DATA_SPLIT = 'test'

    if cfg.DATA_SPLIT == 'train':
        cfg.SUBSET_INDICES = list(range(7))
    elif cfg.DATA_SPLIT == 'test':
        cfg.SUBSET_INDICES = [8, 9]
    else:
        cfg.SUBSET_INDICES = None
    cfg.CASE_INDICES = [0,2]

    volume_generator = luna16_volume_generator.Build_DLP_luna16_volume_generator(
        cfg.RAW_DATA_PATH, subset_indices=cfg.SUBSET_INDICES, case_indices=cfg.CASE_INDICES, only_nodule_slices=cfg.ONLY_NODULES)
    # volume_eval(cfg, volume_generator=volume_generator)
    froc(cfg, volume_generator)
    # CalculateFROC(cfg.DATASET_NAME, cfg.SAVE_PATH)
    # calculateFROC(cfg)
    # eval(cfg)
    return cfg


def asus_malignant_eval():
    cfg = common_config()
    cfg.RAW_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant'
    cfg.DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\malignant\raw'
    cfg = add_dataset_name(cfg)
    cfg.DATA_SPLIT = 'train'

    cfg.SUBSET_INDICES = None
    if cfg.DATA_SPLIT == 'train':
        cfg.CASE_INDICES = list(range(40))
    elif cfg.DATA_SPLIT == 'test':
        cfg.CASE_INDICES = list(range(45, 57))
    else:
        cfg.CASE_INDICES = None

    volume_generator = asus_nodule_volume_generator(cfg.RAW_DATA_PATH, subset_indices=cfg.SUBSET_INDICES, case_indices=cfg.CASE_INDICES,
                                     only_nodule_slices=cfg.ONLY_NODULES)
    volume_eval(cfg, volume_generator=volume_generator)
    return cfg


def asus_benign_eval():
    cfg = common_config()
    cfg.RAW_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\benign'
    cfg.DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\benign\raw'
    cfg = add_dataset_name(cfg)
    cfg.DATA_SPLIT = 'test'

    cfg.SUBSET_INDICES = None
    if cfg.DATA_SPLIT == 'train':
        cfg.CASE_INDICES = list(range(25))
    elif cfg.DATA_SPLIT == 'test':
        cfg.CASE_INDICES = list(range(27, 35))
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
    
    
    