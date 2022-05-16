

import os
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
from sklearn.metrics import confusion_matrix
from evaluation.volume_eval import volumetric_data_eval
from utils.utils import Nodule_data_recording, DataFrameTool, SubmissionDataFrame, irc2xyz, get_nodule_center
from visualization.vis import save_mask, visualize, save_mask_in_3d, plot_scatter, ScatterVisualizer
# import liwei_eval
from evaluationScript import noduleCADEvaluationLUNA16
from postprocessing.reduce_false_positive import NoduleClassifier
from postprocessing.data_postprocess import VolumePostProcessor
from data.volume_generator import luna16_volume_generator, asus_nodule_volume_generator, build_pred_generator, lidc_nodule_volume_generator
from postprocessing.lung_mask_filtering import FalsePositiveReducer
from utils.nodule import LungNoduleStudy
from data.dataloader import GeneralDataset, SimpleNoduleDataset, CropNoduleDataset
from data.data_utils import get_pids_from_coco
from evaluation.evaluator import Pytorch2dSegEvaluator, Pytorch3dSegEvaluator, D2SegEvaluator
from evaluation.keras_evaluator import Keras3dSegEvaluator
from data.volume_to_3d_crop import CropVolume
from model import build_model

from model.d2_model import BatchPredictor
from evaluation.eval_config import get_eval_config
# TODO:
from config import nodule_dataset_config
from utils.configuration import get_device

logging.basicConfig(level=logging.INFO)

# TODO: modify the name of lung_mask_filtering and reduce_false_p


def eval(cfg, volume_generator, data_converter, predictor, evaluator_gen):
    save_path = os.path.join(cfg.SAVE_PATH, cfg.DATASET_NAME, cfg.DATA_SPLIT)
    save_vis_condition = lambda x: True if cfg.SAVE_ALL_IMAGES else True if x < cfg.MAX_SAVE_IMAGE_CASES else False

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
                              lung_mask_path=cfg.LUNG_MASK_PATH, overlapping=cfg.Inference.overlapping, reweight=cfg.Inference.reweight, 
                              reweight_sigma=cfg.Inference.reweight_sigma)
    # evaluator.crop_test_luna16()
    target_studys, pred_studys = evaluator.run()
    return target_studys, pred_studys


def cross_valid_eval():
    cfg = get_eval_config()
    dataset_names = cfg.DATA.NAMES
    model_name = cfg.MODEL.NAME

    model_weight = cfg.MODEL.WEIGHTS
    checkpoint_path = os.path.join(cfg.MODEL.OUTPUT_DIR, cfg.MODEL.WEIGHTS)
    save_path = cfg.SAVE_PATH
    cfg.MODEL_NAME = model_name
    # TODO: move to config
    assign_fold = 4

    if assign_fold is not None:
        assert assign_fold < cfg.EVAL.CV_FOLD, 'Assign fold out of range'
        fold_indices = [assign_fold]
    else:
        fold_indices = list(range(cfg.EVAL.CV_FOLD))

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
            paths = nodule_dataset_config(dataset_name)
            cfg.RAW_DATA_PATH= paths['raw']
            cfg.LUNG_MASK_PATH = paths['lung_mask']

            cfg.DATASET_NAME = dataset_name
            cfg.FOLD = fold
            if cfg.MODEL_NAME in ['2D-Mask-RCNN']:
                cfg.N_CLASS = cfg['d2'].MODEL.ROI_HEADS.NUM_CLASSES
            else:
                cfg.N_CLASS = cfg.DATA.N_CLASS
            cfg.SLICE_SHIFT = cfg.DATA.SLICE_SHIFT
            

            # TODO: exp
            if 'TMH' in dataset_name:
                coco_path = os.path.join(cfg.PATH.DATA_ROOT[dataset_name], 'coco', cfg.TASK_NAME, f'cv-{cfg.EVAL.CV_FOLD}', str(fold))
                case_pids = get_pids_from_coco(os.path.join(coco_path, f'annotations_{cfg.DATA.SPLIT}.json'))
                # case_pids = case_pids[:2]
                volume_generator = asus_nodule_volume_generator(cfg.RAW_DATA_PATH, case_pids=case_pids)
            elif dataset_name == 'LUNA16':
                subset_indices = [1]
                # volume_generator = luna16_volume_generator.Build_DLP_luna16_volume_generator(
                #     data_path=cfg.RAW_DATA_PATH, subset_indices=subset_indices)
                mask_path = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\luna16_mask'
                volume_generator = luna16_volume_generator.Build_LIDC_luna16_volume_generator(
                    data_path=cfg.RAW_DATA_PATH, mask_path=mask_path, subset_indices=subset_indices)
            elif dataset_name == 'LIDC':
                # Convert data to image
                raw_vol_path = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data'
                mask_vol_path = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-preprocess\masks_test\3'
                coco_path = os.path.join(cfg.PATH.DATA_ROOT[dataset_name], 'coco', cfg.TASK_NAME, f'cv-{cfg.EVAL.CV_FOLD}', str(fold))
                case_pids = get_pids_from_coco(os.path.join(coco_path, f'annotations_{cfg.DATA.SPLIT}.json'))
                # case_pids = case_pids[21:]
                volume_generator_builder = lidc_nodule_volume_generator(
                    data_path=raw_vol_path, mask_path=mask_vol_path, case_indices=case_pids)
                volume_generator = volume_generator_builder.build()

            in_planes = 2*cfg.SLICE_SHIFT + 1
            if cfg.MODEL_NAME == '2D-Mask-RCNN':
                data_converter = None
                predictor = BatchPredictor(cfg['d2'])
                evaluator_gen = D2SegEvaluator
            elif cfg.MODEL_NAME in ['2D-FCN', '2D-Unet']:
                data_converter = SimpleNoduleDataset
                predictor = build_model.build_seg_model(model_name=cfg.MODEL_NAME, in_planes=in_planes, n_class=cfg.N_CLASS, 
                                                        device=get_device(), pytorch_pretrained=True, checkpoint_path=checkpoint_path)
                # predictor = build_model.build_seg_model(model_name=cfg.MODEL_NAME, in_planes=3, n_class=2, device=get_device(), pytorch_pretrained=True, checkpoint_path=checkpoint_path)
                evaluator_gen = Pytorch2dSegEvaluator
            elif cfg.MODEL_NAME in ['Model_Genesis', '3D-Unet']:
                # TODO: make the difference between 3D unet nad model_genensis
                data_converter = CropNoduleDataset
                predictor = build_model.build_seg_model(model_name=cfg.MODEL_NAME, in_planes=in_planes, n_class=cfg.N_CLASS, 
                                                        device=get_device(), pytorch_pretrained=True, checkpoint_path=checkpoint_path)
                evaluator_gen = Pytorch3dSegEvaluator
            elif cfg.MODEL_NAME in ['k_Model_Genesis', 'k_3D-Unet']:
                data_converter = CropVolume((64,64,32), (0,0,0), convert_dtype=np.float32, overlapping=0.5)
                predictor = build_model.build_keras_unet3d(
                    cfg.DATA.crop_row, cfg.DATA.crop_col, cfg.DATA.crop_index, checkpoint_path=checkpoint_path)
                evaluator_gen = Keras3dSegEvaluator
            target_studys, pred_studys = eval(cfg, volume_generator, data_converter, predictor, evaluator_gen)


            for target_study, pred_study in zip(target_studys, pred_studys):
                if dataset_name == 'TMH-Benign':
                    benign_target_scatter_vis.record(target_study)
                    benign_pred_scatter_vis.record(pred_study)
                    
                elif dataset_name == 'TMH-Malignant':
                    malignant_target_scatter_vis.record(target_study)
                    malignant_pred_scatter_vis.record(pred_study)

                for target_nodule_id in target_study.nodule_instances:
                    target_nodule = target_study.nodule_instances[target_nodule_id]
                    if dataset_name == 'TMH-Benign':
                        benign_data = [target_study.study_id, 'target', target_nodule.id, target_nodule.hu, target_nodule.nodule_size, 
                                       target_nodule.nodule_center['index'], target_nodule.nodule_center['row'], target_nodule.nodule_center['column'], 
                                       target_nodule.nodule_score['IoU'], target_nodule.nodule_score['DSC'], 0,
                                       target_study.get_score('NoduleTP'), target_study.get_score('NoduleFP'), target_study.get_score('NoduleFN')]
                        benign_rcorder.write_row(benign_data)
                    elif dataset_name == 'TMH-Malignant':
                        malignant_data = [target_study.study_id, 'target', target_nodule.id, target_nodule.hu, target_nodule.nodule_size, 
                                          target_nodule.nodule_center['index'], target_nodule.nodule_center['row'], target_nodule.nodule_center['column'], 
                                          target_nodule.nodule_score['IoU'], target_nodule.nodule_score['DSC'], 0,
                                          target_study.get_score('NoduleTP'), target_study.get_score('NoduleFP'), target_study.get_score('NoduleFN')]
                        malignant_rcorder.write_row(malignant_data)

                for pred_nodule_id in pred_study.nodule_instances:
                    pred_nodule = pred_study.nodule_instances[pred_nodule_id]
                    if dataset_name == 'TMH-Benign':
                        benign_data = [pred_study.study_id, 'pred', pred_nodule.id, pred_nodule.hu, pred_nodule.nodule_size, 
                                       pred_nodule.nodule_center['index'], pred_nodule.nodule_center['row'], pred_nodule.nodule_center['column'], 
                                       pred_nodule.nodule_score['IoU'], pred_nodule.nodule_score['DSC'], 0, 
                                       pred_study.get_score('NoduleTP'), pred_study.get_score('NoduleFP'), pred_study.get_score('NoduleFN')]
                        benign_rcorder.write_row(benign_data)
                    elif dataset_name == 'TMH-Malignant':
                        malignant_data = [pred_study.study_id, 'pred', pred_nodule.id, pred_nodule.hu, pred_nodule.nodule_size, 
                                          pred_nodule.nodule_center['index'], pred_nodule.nodule_center['row'], pred_nodule.nodule_center['column'], 
                                          pred_nodule.nodule_score['IoU'], pred_nodule.nodule_score['DSC'], 0, 
                                          pred_study.get_score('NoduleTP'), pred_study.get_score('NoduleFP'), pred_study.get_score('NoduleFN')]
                        malignant_rcorder.write_row(malignant_data)


    benign_target_scatter_vis.show_scatter(save_path=os.path.join(cfg.MODEL.OUTPUT_DIR, 'benign_target_nodule.png'), 
                                           title='Benign target nodule', xlabel='size (pixels)', ylabel='meanHU')
    benign_pred_scatter_vis.show_scatter(save_path=os.path.join(cfg.MODEL.OUTPUT_DIR, 'benign_pred_nodule.png'), 
                                         title='Benign predict nodule', xlabel='size (pixels)', ylabel='meanHU')
    malignant_target_scatter_vis.show_scatter(save_path=os.path.join(cfg.MODEL.OUTPUT_DIR, 'malignant_target_nodule.png'), 
                                              title='Malignant target nodule', xlabel='size (pixels)', ylabel='meanHU')
    malignant_pred_scatter_vis.show_scatter(save_path=os.path.join(cfg.MODEL.OUTPUT_DIR, 'malignant_pred_nodule.png'), 
                                            title='Malignant predict nodule', xlabel='size (pixels)', ylabel='meanHU')
    
    # TODO: file rename
    benign_rcorder.save_data_frame(save_path=os.path.join(cfg.SAVE_PATH, 'benign.csv'))
    malignant_rcorder.save_data_frame(save_path=os.path.join(cfg.SAVE_PATH, 'malignant.csv'))


def main():
    cross_valid_eval()
    

if __name__ == '__main__':
    main()

    