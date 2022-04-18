import os
from data.volume_generator import asus_nodule_volume_generator
from utils.keras_evaluator import Keras3dSegEvaluator
from data.volume_to_3d_crop import CropVolume
from data.data_utils import get_pids_from_coco
from model import keras_unet3d
from utils.volume_eval import volumetric_data_eval
from data.data_postprocess import VolumePostProcessor
from lung_mask_filtering import FalsePositiveReducer
# from reduce_false_positive import NoduleClassifier
from utils.configuration import load_config


# def xx():
#     for target_study, pred_study in zip(target_studys, pred_studys):
#         if dataset_name == 'ASUS-Benign':
#             benign_target_scatter_vis.record(target_study)
#             benign_pred_scatter_vis.record(pred_study)
            
#         elif dataset_name == 'ASUS-Malignant':
#             malignant_target_scatter_vis.record(target_study)
#             malignant_pred_scatter_vis.record(pred_study)

#         for target_nodule_id in target_study.nodule_instances:
#             target_nodule = target_study.nodule_instances[target_nodule_id]
#             if dataset_name == 'ASUS-Benign':
#                 benign_data = [target_study.study_id, 'target', target_nodule.id, target_nodule.hu, target_nodule.nodule_size, 
#                                 target_nodule.nodule_center['index'], target_nodule.nodule_center['row'], target_nodule.nodule_center['column'], 
#                                 target_nodule.nodule_score['IoU'], target_nodule.nodule_score['DSC'], 0,
#                                 target_study.get_score('NoduleTP'), target_study.get_score('NoduleFP'), target_study.get_score('NoduleFN')]
#                 benign_rcorder.write_row(benign_data)
#             elif dataset_name == 'ASUS-Malignant':
#                 malignant_data = [target_study.study_id, 'target', target_nodule.id, target_nodule.hu, target_nodule.nodule_size, 
#                                     target_nodule.nodule_center['index'], target_nodule.nodule_center['row'], target_nodule.nodule_center['column'], 
#                                     target_nodule.nodule_score['IoU'], target_nodule.nodule_score['DSC'], 0,
#                                     target_study.get_score('NoduleTP'), target_study.get_score('NoduleFP'), target_study.get_score('NoduleFN')]
#                 malignant_rcorder.write_row(malignant_data)

#         for pred_nodule_id in pred_study.nodule_instances:
#             pred_nodule = pred_study.nodule_instances[pred_nodule_id]
#             if dataset_name == 'ASUS-Benign':
#                 benign_data = [pred_study.study_id, 'pred', pred_nodule.id, pred_nodule.hu, pred_nodule.nodule_size, 
#                                 pred_nodule.nodule_center['index'], pred_nodule.nodule_center['row'], pred_nodule.nodule_center['column'], 
#                                 pred_nodule.nodule_score['IoU'], pred_nodule.nodule_score['DSC'], 0, 
#                                 pred_study.get_score('NoduleTP'), pred_study.get_score('NoduleFP'), pred_study.get_score('NoduleFN')]
#                 benign_rcorder.write_row(benign_data)
#             elif dataset_name == 'ASUS-Malignant':
#                 malignant_data = [pred_study.study_id, 'pred', pred_nodule.id, pred_nodule.hu, pred_nodule.nodule_size, 
#                                     pred_nodule.nodule_center['index'], pred_nodule.nodule_center['row'], pred_nodule.nodule_center['column'], 
#                                     pred_nodule.nodule_score['IoU'], pred_nodule.nodule_score['DSC'], 0, 
#                                     pred_study.get_score('NoduleTP'), pred_study.get_score('NoduleFP'), pred_study.get_score('NoduleFN')]
#                 malignant_rcorder.write_row(malignant_data)


#     benign_target_scatter_vis.show_scatter(save_path=os.path.join(cfg.OUTPUT_DIR, 'benign_target_nodule.png'), 
#                                            title='Benign target nodule', xlabel='size (pixels)', ylabel='meanHU')
#     benign_pred_scatter_vis.show_scatter(save_path=os.path.join(cfg.OUTPUT_DIR, 'benign_pred_nodule.png'), 
#                                          title='Benign predict nodule', xlabel='size (pixels)', ylabel='meanHU')
#     malignant_target_scatter_vis.show_scatter(save_path=os.path.join(cfg.OUTPUT_DIR, 'malignant_target_nodule.png'), 
#                                               title='Malignant target nodule', xlabel='size (pixels)', ylabel='meanHU')
#     malignant_pred_scatter_vis.show_scatter(save_path=os.path.join(cfg.OUTPUT_DIR, 'malignant_pred_nodule.png'), 
#                                             title='Malignant predict nodule', xlabel='size (pixels)', ylabel='meanHU')
    
#     # TODO: file rename
#     benign_rcorder.save_data_frame(save_path=os.path.join(cfg.SAVE_PATH, 'benign.csv'))
#     malignant_rcorder.save_data_frame(save_path=os.path.join(cfg.SAVE_PATH, 'malignant.csv'))


def simple_eval(cfg, dataset_name, volume_generator, data_converter, predictor, evaluator_gen):
    save_path = os.path.join(cfg.SAVE_PATH, dataset_name, cfg.DATA_SPLIT)
    save_vis_condition = lambda x: True if cfg.SAVE_ALL_COMPARES else True if x < cfg.MAX_SAVE_IMAGE_CASES else False
    lung_mask_path = os.path.join(cfg.DATA_PATH, 'Lung_Mask_show')

    vol_metric = volumetric_data_eval(
        model_name=cfg.MODEL_NAME, save_path=save_path, dataset_name=dataset_name, match_threshold=cfg.MATCHING_THRESHOLD)

    post_processer = VolumePostProcessor(cfg.connectivity, cfg.area_threshold)
    
    fp_reduce_condition = (cfg.remove_1_slice or cfg.remove_unusual_nodule_by_lung_size or cfg.lung_mask_filtering)
    if fp_reduce_condition:
        fp_reducer = FalsePositiveReducer(_1SR=cfg.remove_1_slice, RUNLS=cfg.remove_unusual_nodule_by_lung_size, LMF=cfg.lung_mask_filtering, 
                                          slice_threshold=cfg.pred_slice_threshold, lung_size_threshold=cfg.lung_size_threshold)
    else:
        fp_reducer = None

    # if cfg.nodule_cls:
    #     crop_range = {'index': cfg.crop_range[0], 'row': cfg.crop_range[1], 'column': cfg.crop_range[2]}
    #     nodule_classifier = NoduleClassifier(crop_range, cfg.FP_reducer_checkpoint, prob_threshold=cfg.NODULE_CLS_PROB)
    # else:
    #     nodule_classifier = None
    evaluator = evaluator_gen(predictor, volume_generator, save_path, data_converter=data_converter, eval_metrics=vol_metric, 
                              slice_shift=cfg.SLICE_SHIFT, save_vis_condition=save_vis_condition, max_test_cases=cfg.MAX_TEST_CASES, 
                              post_processer=post_processer, fp_reducer=fp_reducer, nodule_classifier=None, 
                              lung_mask_path=lung_mask_path)
    target_studys, pred_studys = evaluator.run()
    return target_studys, pred_studys


def config(config_path):
    cfg = load_config(config_path, dict_as_member=True)
    # cfg.DATASET_NAME ='ASUS-Benign'
    # cfg.RAW_DATA_PATH = os.path.join(cfg.PATH.DATA_ROOT[cfg.DATASET_NAME], 'merge')
    # cfg.DATA_PATH = os.path.join(cfg.PATH.DATA_ROOT[cfg.DATASET_NAME], 'image')
    cfg.OUTPUT_DIR = rf'C:\Users\test\Desktop\Leon\Projects\ModelsGenesis\keras\downstream_tasks\models\ncs\run_4'
    cfg.MODEL_WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "Vnet-genesis.h5")

    cfg.lung_mask_filtering = False
    cfg.remove_1_slice = False
    cfg.remove_unusual_nodule_by_lung_size = False
    cfg.remove_unusual_nodule_by_ratio = False
    cfg.lung_size_threshold = 0.4
    cfg.pred_slice_threshold = 1
    cfg.nodule_cls = False
    cfg.crop_range = [48, 48, 48]
    cfg.NODULE_CLS_PROB = 0.75
    cfg.connectivity = 26
    cfg.area_threshold = 8
    cfg.MATCHING_THRESHOLD = 0.1
    cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\run_028\ckpt_best.pth'

    cfg.MAX_SAVE_IMAGE_CASES = 100
    cfg.MAX_TEST_CASES = None
    cfg.ONLY_NODULES = True
    cfg.SAVE_ALL_COMPARES = True

    cfg.SLICE_SHIFT = cfg.DATA.SLICE_SHIFT
    cfg.MODEL_NAME = cfg.MODEL.NAME
    cfg.DATA_SPLIT = 'test'

    run = os.path.split(cfg.OUTPUT_DIR)[1]
    weight = os.path.split(cfg.MODEL_WEIGHTS)[1].split('.')[0]
    # dir_name = ['maskrcnn', f'{run}', f'{weight}', f'{cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}']
    dir_name = ['maskrcnn', f'{run}', f'{weight}']
    FPR_model_code = os.path.split(os.path.split(cfg.FP_reducer_checkpoint)[0])[1]
    dir_name.insert(0, '1SR') if cfg.remove_1_slice else dir_name
    dir_name.insert(0, 'RUNR') if cfg.remove_unusual_nodule_by_ratio else dir_name
    dir_name.insert(0, f'RUNLS_TH{cfg.lung_size_threshold}') if cfg.remove_unusual_nodule_by_lung_size else dir_name
    dir_name.insert(0, 'LMF') if cfg.lung_mask_filtering else dir_name
    dir_name.insert(0, f'NC#{FPR_model_code}') if cfg.nodule_cls else dir_name
    # dir_name.insert(0, str(cfg.INPUT.MIN_SIZE_TEST))
    cfg.SAVE_PATH = os.path.join(cfg.OUTPUT_DIR, '-'.join(dir_name))
    return cfg


def build_keras_unet3d(row, col, index, checkpoint_path):
    predictor = keras_unet3d.unet_model_3d((1, row, col, index), batch_normalization=True)
    print(f"[INFO] Load trained model from {checkpoint_path}")
    predictor.load_weights(checkpoint_path)
    return predictor


def cv_eval(config_path):
    cfg = config(config_path)

    if cfg.EVAL.assign_fold is not None:
        assert cfg.EVAL.assign_fold < cfg.EVAL.CV_FOLD, 'Assign fold out of range'
        fold_indices = [cfg.EVAL.assign_fold]
    else:
        fold_indices = list(range(cfg.EVAL.CV_FOLD))

    for fold in fold_indices:
        eval(cfg, fold)


def eval(cfg, fold):
    data_converter = CropVolume
    predictor = build_keras_unet3d(cfg.DATA.crop_row, cfg.DATA.crop_col, cfg.DATA.crop_index, 
                                   checkpoint_path=cfg.MODEL_WEIGHTS)
    # predictor = build_keras_unet3d(test_cfg.DATA.crop_row, test_cfg.DATA.crop_col, 
    #     test_cfg.DATA.crop_index, checkpoint_path=cfg.MODEL.WEIGHTS)
    evaluator_gen = Keras3dSegEvaluator

    for dataset in cfg.DATA.NAMES:
        cfg.RAW_DATA_PATH = os.path.join(cfg.PATH.DATA_ROOT[dataset], 'merge')
        cfg.DATA_PATH = os.path.join(cfg.PATH.DATA_ROOT[dataset], 'image')
        coco_path = os.path.join(
            cfg.PATH.DATA_ROOT[dataset], 'coco', cfg.TASK_NAME, f'cv-{cfg.EVAL.CV_FOLD}', str(fold))
        case_pids = get_pids_from_coco(os.path.join(coco_path, f'annotations_{cfg.DATA.SPLIT}.json'))
        # TODO:   
        case_pids = case_pids[0:1]

        volume_generator = asus_nodule_volume_generator(cfg.RAW_DATA_PATH, case_pids=case_pids)
        target_studys, pred_studys = simple_eval(
            cfg, dataset, volume_generator, data_converter, predictor, evaluator_gen)
    return target_studys, pred_studys


def main():
    config_path = f'config_file/test.yml'
    cv_eval(config_path)


if __name__ == '__main__':
    main()
    
    # import cv2
    # import matplotlib.pyplot as plt
    # import numpy as np
    # path = rf'C:\Users\test\Desktop\Leon\Projects\ModelsGenesis\keras\downstream_tasks\models\ncs\run_3\maskrcnn-run_3-Vnet-genesis-0,5\ASUS-Malignant\test\images\1m0041\origin\vis-1m0041-76.png'
    
    # img = cv2.imread(path)

    # # steps = [32]
    # # colors = [(255,0,0)]
    # steps = [32, 64]
    # colors = [(255,0,0), (0,0,255)]
    # for step, color in zip(steps, colors):
    #     rows, cols = 512, 512
    #     x = np.arange(0, cols, step)
    #     y = np.arange(0, rows, step)


    #     v_xy = []
    #     h_xy = []
    #     for i in range(len(x)):
    #         v_xy.append( [int(x[i]), 0, int(x[i]), rows-1] )
    #         h_xy.append( [0, int(y[i]), cols-1, int(y[i])] )


    #     for i in range(len(x)):
    #         [x1, y1, x2, y2] = v_xy[i]
    #         [x1_, y1_, x2_, y2_] = h_xy[i]


    #         cv2.line(img, (x1,y1), (x2, y2), color,1 )
    #         cv2.line(img, (x1_,y1_), (x2_, y2_), color,1 )


    # plt.imshow(img)
    # plt.show()