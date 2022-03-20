

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
from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm
import torch

import site_path
from modules.utils import configuration
from utils.utils import cv2_imshow
from config import common_config, dataset_config
from model import build_model
from utils.trainer import Trainer

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


def detectron2_model_train(cfg):
    trainer = DefaultTrainer(cfg) 
    val_loss = ValidationLoss(cfg)  
    trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()


def train(cfg):
    model = build_model(model_name=cfg.MODEL_NAME, slice_shift=cfg.SLICE_SHIFT, n_class=cfg.N_CLASS, pretrained=cfg.isPretrained)
    trainer = Trainer(model)


def main():
    # train_cfg = build_train_config()



    train_cfg = configuration.load_config(f'config_file/train.yml', dict_as_member=True)

    # TODO: combine common, dataset to one function
    cfg = common_config()
    cfg.CV_FOLD = train_cfg.CV_FOLD
    cfg = dataset_config(cfg, train_cfg.DATA.NAMES)
    
    output_dir = cfg.OUTPUT_DIR
    for fold in range(cfg.CV_FOLD):
        train_dataset = tuple([f'{dataset_name}-train-cv{cfg.CV_FOLD}-{fold}' for dataset_name in train_cfg.DATA.NAMES])
        valid_dataset = tuple([f'{dataset_name}-valid-cv{cfg.CV_FOLD}-{fold}' for dataset_name in train_cfg.DATA.NAMES])

        cfg.DATASETS.TRAIN = train_dataset
        cfg.DATASETS.VAL = valid_dataset
        cfg.DATASETS.TEST = ()
        
        cfg.OUTPUT_DIR = os.path.join(output_dir, str(fold))
        if not os.path.isdir(cfg.OUTPUT_DIR):
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        detectron2_model_train(cfg)


if __name__ == '__main__':
    main()