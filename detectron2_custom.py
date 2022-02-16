

import detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import tensorboardX

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
import matplotlib.pyplot as plt
from utils.utils import cv2_imshow
import torch
from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm

import site_path
from modules.utils import train_utils


class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        self._loader = iter(build_detection_train_loader(self.cfg))
        
    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                 **loss_dict_reduced)


def common_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.MODEL.WEIGHTS = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_015\model_0031999.pth' 
    # cfg.MODEL.WEIGHTS = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_032\model_0019999.pth' 
    # cfg.MODEL.WEIGHTS = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_037\model_0015999.pth' 
    # cfg.MODEL.WEIGHTS = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_040\model_0007999.pth' 
    # cfg.MODEL.WEIGHTS = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_052\model_0015999.pth' 
    # cfg.MODEL.WEIGHTS = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_057\model_0009999.pth' 
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00005  
    cfg.SOLVER.MAX_ITER = 20000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.OUTPUT_DIR = train_utils.create_training_path('output')
    # # By default, {MIN,MAX}_SIZE options are used in transforms.ResizeShortestEdge.
    # Please refer to ResizeShortestEdge for detailed definition.
    # Size of the smallest side of the image during training
    cfg.INPUT.MIN_SIZE_TRAIN = (512, 800)
    # cfg.INPUT.MIN_SIZE_TRAIN = (640, 800)
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
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "giou"
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "giou"
    # cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4,  8,  16,  32,  64]]
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8,  16,  32,  64, 128]]
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.2]]
    # cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 5e-3
    # cfg.MODEL.RPN.LOSS_WEIGHT = 5e-3
    # cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 5e-3
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 20
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 20
    # cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    return cfg


def asus_benign_config():
    dataset_name = 'ASUS-Benign'
    cfg = common_config()
    cfg.USING_DATASET.append(dataset_name)
    cfg.TRAIN_JSON_FILE = os.path.join("Annotations", "ASUS_Nodule", "benign", "annotations_train.json")
    cfg.VALID_JSON_FILE = os.path.join("Annotations", "ASUS_Nodule", "benign", "annotations_valid.json")
    cfg.DATA_PATH = rf"C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\benign\raw"
    # Prepare the dataset
    register_coco_instances(f"{dataset_name}-train", {}, cfg.TRAIN_JSON_FILE, cfg.DATA_PATH)
    register_coco_instances(f"{dataset_name}-valid", {}, cfg.VALID_JSON_FILE, cfg.DATA_PATH)
    return cfg


def asus_malignant_config():
    dataset_name = 'ASUS-Malignant'
    cfg = common_config()
    cfg.USING_DATASET.append(dataset_name)
    cfg.TRAIN_JSON_FILE = os.path.join("Annotations", "ASUS_Nodule", "malignant", "annotations_train.json")
    cfg.VALID_JSON_FILE = os.path.join("Annotations", "ASUS_Nodule", "malignant", "annotations_valid.json")
    cfg.DATA_PATH = rf"C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\malignant\raw"
    # Prepare the dataset
    register_coco_instances(f"{dataset_name}-train", {}, cfg.TRAIN_JSON_FILE, cfg.DATA_PATH)
    register_coco_instances(f"{dataset_name}-valid", {}, cfg.VALID_JSON_FILE, cfg.DATA_PATH)
    return cfg


def luna16_config():
    dataset_name = 'LUNA16'
    cfg = common_config()
    cfg.USING_DATASET.append(dataset_name)
    cfg.TRAIN_JSON_FILE = os.path.join("Annotations", dataset_name, "annotations_train.json")
    cfg.VALID_JSON_FILE = os.path.join("Annotations", dataset_name, "annotations_valid.json")
    cfg.DATA_PATH = rf"C:\Users\test\Desktop\Leon\Datasets\{dataset_name}-preprocess\raw"
    # Prepare the dataset
    register_coco_instances(f"{dataset_name}-train", {}, cfg.TRAIN_JSON_FILE, cfg.DATA_PATH)
    register_coco_instances(f"{dataset_name}-valid", {}, cfg.VALID_JSON_FILE, cfg.DATA_PATH)
    return cfg


def luna16_round_config():
    cfg = common_config()
    cfg.TRAIN_JSON_FILE = os.path.join("Annotations", "LUNA16-round", "annotations_train.json")
    cfg.VALID_JSON_FILE = os.path.join("Annotations", "LUNA16-round", "annotations_valid.json")
    cfg.DATA_PATH = rf"C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess-round\raw"
    return cfg


def lidc_config():
    cfg = common_config()
    cfg.TRAIN_JSON_FILE = os.path.join("Annotations", "LIDC", "annotations_train.json")
    cfg.VALID_JSON_FILE = os.path.join("Annotations", "LIDC", "annotations_valid.json")
    cfg.DATA_PATH = rf"C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-Preprocessing-png\Image"
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    return cfg


def dataset_config(cfg, using_dataset):
    cfg.USING_DATASET = ''
    for dataset_name in using_dataset:
        train_json = os.path.join("Annotations", dataset_name, "annotations_train.json")
        valid_json = os.path.join("Annotations", dataset_name, "annotations_valid.json")
        datra_path = rf"C:\Users\test\Desktop\Leon\Datasets\{dataset_name}-preprocess\raw"
        # Prepare the dataset
        register_coco_instances(f"{dataset_name}-train", {}, train_json, datra_path)
        register_coco_instances(f"{dataset_name}-valid", {}, valid_json, datra_path)
    return cfg


def main():
    cfg = common_config()
    using_dataset = ['ASUS-Benign', 'ASUS-Malignant'] # 'LUNA16', 'ASUS-Benign', 'ASUS-Malignant'
    cfg = dataset_config(cfg, using_dataset)

    train_dataset = tuple([f'{dataset_name}-train' for dataset_name in using_dataset])
    valid_dataset = tuple([f'{dataset_name}-valid' for dataset_name in using_dataset])

    cfg.DATASETS.TRAIN = train_dataset
    cfg.DATASETS.VAL = valid_dataset
    cfg.DATASETS.TEST = ()

    # metadata = MetadataCatalog.get("my_dataset_train")

    # dataset_dicts = DatasetCatalog.get("my_dataset_train")
    # for image_idx, d in enumerate(random.sample(dataset_dicts, 3)):
    # # for image_idx, d in enumerate(dataset_dicts[:87]):
    # # for image_idx, d in enumerate(dataset_dicts):
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img, metadata=metadata, scale=1.0)
    #     out = visualizer.draw_dataset_dict(d)
    #     cv2_imshow(out.get_image(), os.path.join(cfg.OUTPUT_DIR, f'input_samples_{image_idx:03d}.png'))

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    val_loss = ValidationLoss(cfg)  
    trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == '__main__':
    main()