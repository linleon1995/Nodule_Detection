

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
from config import build_d2_config, d2_register_coco, build_train_config
from model import build_model
from utils.trainer import Trainer
from utils import train_utils
from data import data_utils
from data import dataloader


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


def d2_model_train(train_cfg):
    cfg = train_cfg['d2']
    d2_register_coco(cfg, train_cfg.DATA.NAMES.keys())

    assign_fold = 4
    if assign_fold is not None:
        assert assign_fold < train_cfg.CV_FOLD, 'Assign fold out of range'
        fold_indices = [assign_fold]
    else:
        fold_indices = list(range(train_cfg.CV_FOLD))
    
    output_dir = cfg.OUTPUT_DIR
    for fold in fold_indices:
        train_dataset = tuple([f'{dataset_name}-train-cv{cfg.CV_FOLD}-{fold}' for dataset_name in train_cfg.DATA.NAMES.keys()])
        valid_dataset = tuple([f'{dataset_name}-valid-cv{cfg.CV_FOLD}-{fold}' for dataset_name in train_cfg.DATA.NAMES.keys()])

        cfg.DATASETS.TRAIN = train_dataset
        cfg.DATASETS.VAL = valid_dataset
        cfg.DATASETS.TEST = ()
        
        cfg.OUTPUT_DIR = os.path.join(output_dir, str(fold))
        if not os.path.isdir(cfg.OUTPUT_DIR):
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        trainer = DefaultTrainer(cfg) 
        val_loss = ValidationLoss(cfg)  
        trainer.register_hooks([val_loss])
        # swap the order of PeriodicWriter and ValidationLoss
        trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
        trainer.resume_or_load(resume=False)
        trainer.train()
    

def pytorch_model_train(cfg):
    exp_path = train_utils.create_training_path('checkpoints')
    checkpoint_path = cfg.TRAIN.CHECKPOINT_PATH
    model = build_model.build_seg_model(model_name=cfg.MODEL.NAME, slice_shift=cfg.DATA.SLICE_SHIFT, n_class=cfg.DATA.N_CLASS, device=configuration.get_device())

    # TODO: annotation json and cv
    train_cases = data_utils.get_pids_from_coco(
        [os.path.join(cfg.DATA.NAMES[dataset_name]['COCO_PATH'], f'annotations_train.json') for dataset_name in cfg.DATA.NAMES])
    valid_cases = data_utils.get_pids_from_coco(
        [os.path.join(cfg.DATA.NAMES[dataset_name]['COCO_PATH'], f'annotations_test.json') for dataset_name in cfg.DATA.NAMES])

    # TODO: try to use dataset config
    input_roots = [rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodule\ASUS-Malignant\shift\3\input',
                   rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodule\ASUS-Benign\shift\3\input']
    target_roots = [rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodule\ASUS-Malignant\shift\3\target',
                    rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodule\ASUS-Benign\shift\3\target']

    train_dataloader, valid_dataloader = dataloader.build_dataloader(input_roots, target_roots, train_cases, valid_cases, cfg.DATA.BATCH_SIZE)
    loss = train_utils.create_criterion(cfg.TRAIN.LOSS, n_class=cfg.DATA.N_CLASS)
    optimizer = train_utils.create_optimizer(lr=cfg.TRAIN.LR, optimizer_config=cfg.TRAIN.OPTIMIZER, model=model)
    valid_activation = train_utils.create_activation(cfg.VALID.ACTIVATION)

    # Logger
    logger = train_utils.get_logger('train')
    logger.info('Start Training!!')
    logger.info(f'Training epoch: {cfg.TRAIN.EPOCH} Batch size: {cfg.DATA.BATCH_SIZE} Training Samples: {len(train_dataloader.dataset)}')
    train_utils.config_logging(os.path.join(exp_path, 'logging.txt'), cfg, access_mode='w+')

    trainer = Trainer(model,
                      criterion=loss,
                      optimizer=optimizer,
                      train_dataloader=train_dataloader,
                      valid_dataloader=valid_dataloader,
                      logger=logger,
                      device=configuration.get_device(),
                      n_class=cfg.DATA.N_CLASS,
                      exp_path=exp_path,
                      train_epoch=cfg.TRAIN.EPOCH,
                      batch_size=cfg.DATA.BATCH_SIZE,
                      valid_activation=valid_activation,
                      history=checkpoint_path)
    trainer.fit()


def main():
    config_path = f'config_file/train.yml'
    train_cfg = build_train_config(config_path)

    if train_cfg.MODEL.NAME in ['2D-Mask-RCNN']:
        d2_model_train(train_cfg)
    else:
        pytorch_model_train(train_cfg)




if __name__ == '__main__':
    main()