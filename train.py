

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
from utils.utils import cv2_imshow
from config import common_config, dataset_config

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


def main():
    cfg = common_config()
    using_dataset = ['LUNA16'] # 'LUNA16', 'ASUS-Benign', 'ASUS-Malignant'
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