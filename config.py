import os
import detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.config import get_cfg

import site_path
from utils import configuration
from utils.train_utils import create_training_path


class models_genesis_config:
    model = "Unet3D"
    suffix = "genesis_chest_ct"
    exp_name = model + "-" + suffix
    
    # data
    data = "generated_cubes"
    # train_fold=[0]
    train_fold=[0,1]
    valid_fold=[5]
    test_fold=[7,8,9]
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 32
    input_rows = 64
    input_cols = 64 
    input_deps = 32
    nb_class = 1
    
    # model pre-training
    verbose = 1
    weights = rf'C:\Users\test\Desktop\Leon\Projects\ModelsGenesis\pretrained_weights\Genesis_Chest_CT.pt'
    # weights = 'pretrained_weights/Unet3D-genesis_chest_ct/run_015/ckpt-best.pt'
    # weights = None
    batch_size = 3
    optimizer = "sgd"
    workers = 10
    max_queue_size = workers * 4
    save_samples = "png"
    nb_epoch = 200
    patience = 50
    lr = 5e-4
    remove_empty_sample = False
    class_balance = False
    annot_path = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data\annotations.csv'

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4
    
    # logs
    model_path = "pretrained_weights"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logs_path = os.path.join(model_path, "Logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")



def nodule_dataset_config(name):
    TMH_Benign = {
        'raw': rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\TMH-Benign\merge',
        'lung_mask': rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\TMH-Benign\image\Lung_Mask_show'
    }

    TMH_Malignant = {
        'raw': rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\TMH-Malignant\merge',
        'lung_mask': rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\TMH-Malignant\image\Lung_Mask_show'
    }

    LUNA16 = {
        'raw': rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data',
        'lung_mask': None
    }

    dataset_config = {
        'TMH-Benign': TMH_Benign,
        'TMH-Malignant': TMH_Malignant,
        'LUNA16': LUNA16,
    }
    assert name in dataset_config, 'Unknown dataset.'

    return dataset_config[name]


def build_train_config(config_path):
    train_cfg = build_custom_config(config_path)
    if train_cfg.MODEL.backend == 'd2':
        d2_cfg = build_d2_config()
        d2_cfg.CV_FOLD = train_cfg.CV.FOLD
        train_cfg['d2'] = d2_cfg
    return train_cfg


def build_custom_config(config_path):
    custom_cfg = configuration.load_config(config_path, dict_as_member=True)
    return custom_cfg


def build_d2_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00005  
    cfg.SOLVER.MAX_ITER = 8000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.MODEL.WEIGHTS = get_model_weight()
    cfg.OUTPUT_DIR = create_training_path('output')
    # # By default, {MIN,MAX}_SIZE options are used in transforms.ResizeShortestEdge.
    # Please refer to ResizeShortestEdge for detailed definition.
    # Size of the smallest side of the image during training
    cfg.INPUT.MIN_SIZE_TRAIN = (1024, 1600)
    # cfg.INPUT.MIN_SIZE_TRAIN = (800, 1240)
    # Sample size of smallest side by choice or random selection from range give by
    # INPUT.MIN_SIZE_TRAIN
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "range"
    # Maximum size of the side of the image during training
    cfg.INPUT.MAX_SIZE_TRAIN = 1600
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
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
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


def d2_register_coco(cfg, using_dataset):
    for fold in range(cfg.CV_FOLD):
        for dataset_name in using_dataset:
            data_cfg = configuration.load_config(f'data/config/{dataset_name}.yml', dict_as_member=True)
            data_root = data_cfg.PATH.DATA_ROOT
            task_name = data_cfg.TASK_NAME

            # TODO:
            data_root = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\TMH-Malignant'
            train_json = os.path.join(data_root, "coco", task_name, f'cv-{cfg.CV_FOLD}', str(fold), "annotations_train.json")
            valid_json = os.path.join(data_root, "coco", task_name, f'cv-{cfg.CV_FOLD}', str(fold), "annotations_test.json")
            
            # Prepare the dataset
            register_coco_instances(f"{dataset_name}-train-cv{cfg.CV_FOLD}-{fold}", {}, train_json, data_root)
            register_coco_instances(f"{dataset_name}-valid-cv{cfg.CV_FOLD}-{fold}", {}, valid_json, data_root)