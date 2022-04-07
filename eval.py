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
from model import build_model
logging.basicConfig(level=logging.INFO)

import site_path
from modules.utils import configuration

from Liwei.LUNA16_test import util
from Liwei.FTP1m_test import test
import cc3d

# TODO: modify the name of lung_mask_filtering and reduce_false_p



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


def eval(cfg, volume_generator, data_converter, predictor, evaluator_gen):
    save_path = os.path.join(cfg.SAVE_PATH, cfg.DATASET_NAME, cfg.DATA_SPLIT)
    save_vis_condition = lambda x: True if cfg.SAVE_ALL_COMPARES else True if x < cfg.MAX_SAVE_IMAGE_CASES else False
    lung_mask_path = os.path.join(cfg.DATA_PATH, 'Lung_Mask_show')

    vol_metric = volumetric_data_eval(
        model_name=cfg.MODEL_NAME, save_path=save_path, dataset_name=cfg.DATASET_NAME, match_threshold=cfg.MATCHING_THRESHOLD)

    post_processer = VolumePostProcessor(cfg.connectivity, cfg.area_threshold)
    
    fp_reduce_condition = (cfg.remove_1_slice or cfg.remove_unusual_nodule_by_lung_size or cfg.lung_mask_filtering)
    if fp_reduce_condition:
        fp_reducer = FalsePositiveReducer(_1SR=cfg.remove_1_slice, RUNLS=cfg.remove_unusual_nodule_by_lung_size, LMF=cfg.lung_mask_filtering, 
                                          slice_threshold=cfg.pred_slice_threshold, lung_size_threshold=cfg.lung_size_threshold)
    else:
        fp_reducer = None

    if cfg.nodule_cls:
        crop_range = {'index': cfg.crop_range[0], 'row': cfg.crop_range[1], 'column': cfg.crop_range[2]}
        nodule_classifier = NoduleClassifier(crop_range, cfg.FP_reducer_checkpoint, prob_threshold=cfg.NODULE_CLS_PROB)
    else:
        nodule_classifier = None
    evaluator = evaluator_gen(predictor, volume_generator, save_path, data_converter=data_converter, eval_metrics=vol_metric, 
                              slice_shift=cfg.SLICE_SHIFT, save_vis_condition=save_vis_condition, max_test_cases=cfg.MAX_TEST_CASES, 
                              post_processer=post_processer, fp_reducer=fp_reducer, nodule_classifier=nodule_classifier, 
                              lung_mask_path=lung_mask_path)
    target_studys, pred_studys = evaluator.run()
    return target_studys, pred_studys



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
    checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_022'
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


    cfg.OUTPUT_DIR = rf'C:\Users\test\Desktop\Leon\Projects\ModelsGenesis\pretrained_weights\Unet3D-genesis_chest_ct\run_000'
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "ckpt-020.pt")  # path to the model we just trained
    # # cfg.OUTPUT_DIR = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\run_004'
    # # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "ckpt_best.pth")  # path to the model we just trained
    # # cfg.OUTPUT_DIR = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\liwei'
    # # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best.pt")  # path to the model we just trained
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
    cfg.MAX_SAVE_IMAGE_CASES = 100
    cfg.MAX_TEST_CASES = None
    cfg.ONLY_NODULES = True
    cfg.SAVE_ALL_COMPARES = True
    cfg.TEST_BATCH_SIZE = 2
    cfg.SAVE_ALL_IMAGES = True

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
    assign_fold = 0

    if assign_fold is not None:
        assert assign_fold < test_cfg.CV_FOLD, 'Assign fold out of range'
        fold_indices = [assign_fold]
    else:
        fold_indices = list(range(test_cfg.CV_FOLD))

    benign_target_scatter_vis = ScatterVisualizer()
    benign_pred_scatter_vis = ScatterVisualizer()
    malignant_target_scatter_vis = ScatterVisualizer()
    malignant_pred_scatter_vis = ScatterVisualizer()
    # TODO: bug if the value didn't exist
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
            if cfg.MODEL_NAME in ['2D-Mask-RCNN']:
                cfg.N_CLASS = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            else:
                cfg.N_CLASS = test_cfg.DATA.N_CLASS
            cfg.SLICE_SHIFT = test_cfg.DATA.SLICE_SHIFT
            
            # cfg.MODEL.WEIGHTS = os.path.join(os.path.split(model_weight)[0], str(cfg.FOLD), os.path.split(model_weight)[1])
            # cfg.MODEL.WEIGHTS = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\Liwei\FTP1m_test\model\FCN_all_best.pt'

            # cfg.SAVE_PATH = os.path.join(os.path.split(save_path)[0], str(cfg.FOLD), os.path.split(save_path)[1])
            # cfg.SAVE_PATH = os.path.join(os.path.split(cfg.MODEL.WEIGHTS)[0])

            volume_generator = asus_nodule_volume_generator(cfg.RAW_DATA_PATH, 
                                                            case_pids=case_pids)

            in_planes = 2*cfg.SLICE_SHIFT + 1
            if cfg.MODEL_NAME == '2D-Mask-RCNN':
                data_converter = None
                predictor = BatchPredictor(cfg)
                evaluator_gen = D2SegEvaluator
            elif cfg.MODEL_NAME in ['2D-FCN', '2D-Unet']:
                data_converter = SimpleNoduleDataset
                predictor = build_model.build_seg_model(model_name=cfg.MODEL_NAME, in_planes=in_planes, n_class=cfg.N_CLASS, 
                                                        device=configuration.get_device(), pytorch_pretrained=True, checkpoint_path=cfg.MODEL.WEIGHTS)
                # predictor = build_model.build_seg_model(model_name=cfg.MODEL_NAME, in_planes=3, n_class=2, device=configuration.get_device(), pytorch_pretrained=True, checkpoint_path=cfg.MODEL.WEIGHTS)
                evaluator_gen = Pytorch2dSegEvaluator
            elif cfg.MODEL_NAME in ['Model_Genesis', '3D-Unet']:
                # TODO: make the difference between 3D unet nad model_genensis
                data_converter = CropNoduleDataset
                predictor = build_model.build_seg_model(model_name=cfg.MODEL_NAME, in_planes=in_planes, n_class=cfg.N_CLASS, 
                                                        device=configuration.get_device(), pytorch_pretrained=True, checkpoint_path=cfg.MODEL.WEIGHTS)
                evaluator_gen = Pytorch3dSegEvaluator
            target_studys, pred_studys = eval(cfg, volume_generator, data_converter, predictor, evaluator_gen)


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

    # x_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant\crop\32x64x64-1\positive\Image\asus-0037-1m0033.npy'
    # m_path = x_path.replace('Image', 'Mask')
    # x_arr = np.load(x_path)
    # m_arr = np.load(m_path)
    # for s in range(0, x_arr.shape[0], 4):
    #     plt.imshow(x_arr[s], 'gray')
    #     plt.imshow(m_arr[s], alpha=0.2)
    #     plt.show()


    # x_path = rf'C:\Users\test\Desktop\Leon\Projects\ModelsGenesis\generated_cubes\x_test_64x64x32.npy'
    # m_path = rf'C:\Users\test\Desktop\Leon\Projects\ModelsGenesis\generated_cubes\m_test_64x64x32.npy'
    # x_arr = np.load(x_path)
    # m_arr = np.load(m_path)

    # for i in range(x_arr.shape[0]):
    #     for s in range(0, x_arr.shape[3], 4):
    #         if np.sum(m_arr[i,...,s,0])<=0:
    #             plt.imshow(x_arr[i,...,s,0], 'gray')
    #             plt.imshow(m_arr[i,...,s,0], alpha=0.2)
    #             plt.show()


    # path = rf'C:\Users\test\Desktop\Leon\Projects\ModelsGenesis\generated_cubes\bat_32_s_64x64x32_0.npy'
    # arr = np.load(path)
    # for i in range(arr.shape[0]):
    #     for s in range(0, arr.shape[3], 4):
    #         plt.imshow(arr[i,...,s], 'gray')
    #         plt.show()
    
    