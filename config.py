import os
import detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.config import get_cfg

import site_path
from modules.utils import configuration
from modules.utils import train_utils


def get_model_weight():
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_015' 
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_032' 
    checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_037' 
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_040' 
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_052' 
    # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_057' 

    model_weight = os.path.join(checkpoint_path, "model_0015999.pth")  # path to the model we just trained
    # model_weight = os.path.join(checkpoint_path, "model_final.pth")  # path to the model we just trained

    # model_weight = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    return model_weight


def common_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00005  
    cfg.SOLVER.MAX_ITER = 6000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.MODEL.WEIGHTS = get_model_weight()
    cfg.OUTPUT_DIR = train_utils.create_training_path('output')
    # # By default, {MIN,MAX}_SIZE options are used in transforms.ResizeShortestEdge.
    # Please refer to ResizeShortestEdge for detailed definition.
    # Size of the smallest side of the image during training
    cfg.INPUT.MIN_SIZE_TRAIN = (512, 800)
    # cfg.INPUT.MIN_SIZE_TRAIN = (800, 1240)
    # Sample size of smallest side by choice or random selection from range give by
    # INPUT.MIN_SIZE_TRAIN
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "range"
    # Maximum size of the side of the image during training
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    # Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
    # cfg.INPUT.MIN_SIZE_TEST = 800
    # Maximum size of the side of the image during testing
    # cfg.INPUT.MAX_SIZE_TEST = 1333
    # Mode for flipping images used in data augmentation during training
    # choose one of ["horizontal, "vertical", "none"]
    # cfg.INPUT.RANDOM_FLIP = "none"
    cfg.INPUT.RANDOM_FLIP = "horizontal"

    # `True` if cropping is used for data augmentation during training
    cfg.INPUT.CROP.ENABLED = True
    # Cropping type. See documentation of `detectron2.data.transforms.RandomCrop` for explanation.
    # cfg.INPUT.CROP.TYPE = "relative_range"
    cfg.INPUT.CROP.TYPE = "relative_range"
    # Size of crop in range (0, 1] if CROP.TYPE is "relative" or "relative_range" and in number of
    # pixels if CROP.TYPE is "absolute"
    cfg.INPUT.CROP.SIZE = [0.7, 0.7]
    cfg.SOLVER.CHECKPOINT_PERIOD = 3000
    # cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "giou"
    # cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "giou"
    # cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4,  8,  16,  32,  64]]
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8,  16,  32,  64, 128]]
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.2]]
    # cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 5e-3
    # cfg.MODEL.RPN.LOSS_WEIGHT = 5e-3
    # cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 5e-3
    # cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 20
    # cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 20
    # cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    return cfg


def dataset_config(cfg, using_dataset):
    cfg.USING_DATASET = ''

    for fold in range(cfg.CV_FOLD):
        for dataset_name in using_dataset:
            config_name = f'ASUS-{dataset_name}'
            data_cfg = configuration.load_config(f'data/config/{dataset_name}.yml', dict_as_member=True)
            data_root = data_cfg.PATH.DATA_ROOT
            task_name = data_cfg.TASK_NAME

            train_json = os.path.join(data_root, "coco", task_name, f'cv-{cfg.CV_FOLD}', str(fold), "annotations_train.json")
            valid_json = os.path.join(data_root, "coco", task_name, f'cv-{cfg.CV_FOLD}', str(fold), "annotations_test.json")
            
            # Prepare the dataset
            register_coco_instances(f"{dataset_name}-train-cv{cfg.CV_FOLD}-{fold}", {}, train_json, data_root)
            register_coco_instances(f"{dataset_name}-valid-cv{cfg.CV_FOLD}-{fold}", {}, valid_json, data_root)
    return cfg


def get_train_config():
    pass


def build_test_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg = select_model(cfg)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    cfg.MATCHING_THRESHOLD = 0.1
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
    cfg.reduce_false_positive = True
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

    cfg.lung_mask_filtering = True
    cfg.remove_1_slice = True
    cfg.remove_unusual_nodule_by_ratio = False
    cfg.remove_unusual_nodule_by_lung_size = True
    cfg.lung_size_threshold = 0.4
    
    run = os.path.split(cfg.OUTPUT_DIR)[1]
    weight = os.path.split(cfg.MODEL.WEIGHTS)[1].split('.')[0]
    # cfg.SAVE_PATH = rf'C:\Users\test\Desktop\Leon\Weekly\1227'
    dir_name = ['maskrcnn', f'{run}', f'{weight}', f'{cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}']
    FPR_model_code = os.path.split(os.path.split(cfg.FP_reducer_checkpoint)[0])[1]
    dir_name.insert(0, '1SR') if cfg.remove_1_slice else dir_name
    dir_name.insert(0, 'RUNR') if cfg.remove_unusual_nodule_by_ratio else dir_name
    dir_name.insert(0, f'RUNLS_TH{cfg.lung_size_threshold}') if cfg.remove_unusual_nodule_by_lung_size else dir_name
    dir_name.insert(0, 'LMF') if cfg.lung_mask_filtering else dir_name
    dir_name.insert(0, f'NC#{FPR_model_code}') if cfg.reduce_false_positive else dir_name
    dir_name.insert(0, str(cfg.INPUT.MIN_SIZE_TEST))
    cfg.SAVE_PATH = os.path.join(cfg.OUTPUT_DIR, '-'.join(dir_name))
    cfg.MAX_SAVE_IMAGE_CASES = 100
    cfg.MAX_TEST_CASES = None
    cfg.ONLY_NODULES = True
    cfg.SAVE_ALL_COMPARES = False
    cfg.TEST_BATCH_SIZE = 2

    

        # cfg.SUBSET_INDICES = None
        # if cfg.DATA_SPLIT == 'train':
        #     # cfg.CASE_INDICES = list(range(25))
        #     cfg.CASE_INDICES = list(range(17))
        # elif cfg.DATA_SPLIT == 'valid':
        #     # cfg.CASE_INDICES = list(range(25, 27))
        #     cfg.CASE_INDICES = list(range(17, 19))
        # elif cfg.DATA_SPLIT == 'test':
        #     # cfg.CASE_INDICES = list(range(27, 35))
        #     cfg.CASE_INDICES = list(range(19, 25))
        #     # cfg.CASE_INDICES = [20]
        # else:
        #     cfg.CASE_INDICES = None

    return cfg