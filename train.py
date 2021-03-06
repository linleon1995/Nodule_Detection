
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
from torch.utils import tensorboard
# import tensorboardX
import torch

from detectron2.engine import DefaultTrainer
from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm

from utils.configuration import get_device
from utils.train_utils import set_deterministic
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
    if train_cfg.CV.ASSIGN_FOLD is not None:
        assert train_cfg.CV.ASSIGN_FOLD < train_cfg.CV.FOLD, 'Assign fold out of range'
        fold_indices = [train_cfg.CV.ASSIGN_FOLD]
    else:
        fold_indices = list(range(train_cfg.CV.FOLD))

    for dataset_name in train_cfg.DATA.NAMES.keys():
        data_root = train_cfg.DATA[dataset_name]['DATA_ROOT']
        task_name = train_cfg.DATA[dataset_name]['TASK_NAME']
        d2_register_coco(
            train_cfg.CV.FOLD, fold_indices, dataset_name, data_root, task_name)
    
    output_dir = cfg.OUTPUT_DIR
    for fold in fold_indices:
        train_dataset = tuple(
            [f'{dataset_name}-train-cv{train_cfg.CV.FOLD}-{fold}' 
            for dataset_name in train_cfg.DATA.NAMES.keys()]
        )
        valid_dataset = tuple(
            [f'{dataset_name}-valid-cv{train_cfg.CV.FOLD}-{fold}' 
            for dataset_name in train_cfg.DATA.NAMES.keys()]
        )

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
    in_planes = 2*cfg.DATA.SLICE_SHIFT + 1
    model = build_model.build_seg_model(model_name=cfg.MODEL.NAME, in_planes=in_planes, n_class=cfg.DATA.N_CLASS, device=get_device())
    transform_config = cfg.DATA.TRANSFORM

    # TODO: annotation json and cv
    train_cases = data_utils.get_pids_from_coco(
        [os.path.join(cfg.DATA.NAMES[dataset_name]['COCO_PATH'], f'annotations_train.json') for dataset_name in cfg.DATA.NAMES])
    valid_cases = data_utils.get_pids_from_coco(
        [os.path.join(cfg.DATA.NAMES[dataset_name]['COCO_PATH'], f'annotations_test.json') for dataset_name in cfg.DATA.NAMES])

    # TODO: try to use dataset config
    input_roots = [rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodule\TMH-Malignant\shift\3\input',
                   rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodule\TMH-Benign\shift\3\input']
    target_roots = [rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodule\TMH-Malignant\shift\3\target',
                    rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodule\TMH-Benign\shift\3\target']

    train_dataloader, valid_dataloader = dataloader.build_dataloader(
        input_roots, target_roots, train_cases, valid_cases, cfg.DATA.BATCH_SIZE, transform_config=transform_config)
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
                      device=get_device(),
                      n_class=cfg.DATA.N_CLASS,
                      exp_path=exp_path,
                      train_epoch=cfg.TRAIN.EPOCH,
                      batch_size=cfg.DATA.BATCH_SIZE,
                      valid_activation=valid_activation,
                      history=checkpoint_path,
                      checkpoint_saving_steps=cfg.TRAIN.CHECKPOINT_SAVING_STEPS)
    trainer.fit()


def main():
    config_path = f'config_file/train.yml'
    train_cfg = build_train_config(config_path)
    set_deterministic(train_cfg.SEED)

    if train_cfg.MODEL.backend == 'd2':
        d2_model_train(train_cfg)
    elif train_cfg.MODEL.backend == 'pytorch':
        pytorch_model_train(train_cfg)


if __name__ == '__main__':
    main()


    