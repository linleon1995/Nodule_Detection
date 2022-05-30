import os
import random
import numpy as np
import torch
manual_seed = 1
random.seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from model.ResNet_3d import build_3d_resnet
import tensorboardX
from pprint import pprint

from nodule_classification.data.build_crop_dataset import build_dataset, build_coco_path
from utils import trainer
from utils import configuration
from utils import train_utils
from data.data_utils import get_pids_from_coco
from utils.train_utils import create_training_path, set_deterministic


CONFIG_PATH = 'config_file/train_config.yml'
LOGGER = train_utils.get_logger('train')


def main(config_reference):
    # Configuration
    cfg = configuration.load_config(config_reference)
    cfg = train_utils.DictAsMember(cfg)
    device = configuration.get_device(cfg.get('device', None))
    cfg.device = device

    exp_path = create_training_path(cfg.TRAIN.CHECKPOINT_PATH, make_dir=False)
    coco_list = build_coco_path(cfg.DATA.COCO_PATH[cfg.DATA.NAME], cfg.CV.FOLD, cfg.CV.ASSIGN)
    for fold, (train_coco, valid_coco) in enumerate(coco_list):
        cv_exp_path = os.path.join(exp_path, str(fold))
        os.makedirs(cv_exp_path, exist_ok=True)
        train(cfg, train_coco, valid_coco, cv_exp_path)  


def train(cfg, train_coco, valid_coco, exp_path):
    # # Set deterministic
    # # TODO: Set deterministic is not working
    # manual_seed = cfg.TRAIN.manual_seed
    # if manual_seed is not None:
    #     LOGGER.info(f'Seed the RNG for all devices with {manual_seed}')
    #     # train_utils.set_deterministic(manual_seed)
    #     train_utils.set_deterministic(manual_seed, random, np, torch)

    model = build_3d_resnet(
        model_depth=cfg.MODEL.DEPTH, n_classes=cfg.MODEL.NUM_CLASSES, conv1_t_size=7, conv1_t_stride=2)

    # Build dataset
    train_seriesuid = get_pids_from_coco(train_coco)
    valid_seriesuid = get_pids_from_coco(valid_coco)
    train_dataloader, valid_dataloader = build_dataset(
        cfg.DATA.DATA_PATH[cfg.DATA.NAME], cfg.DATA.CROP_RANGE, train_seriesuid, valid_seriesuid,
        cfg.DATA.IS_DATA_AUGMENTATION, cfg.DATA.BATCH_SIZE)

    # Logger
    LOGGER.info("Start Training!!")
    exp_info = f"Training epoch: {cfg.TRAIN.EPOCH} Batch size: {cfg.DATA.BATCH_SIZE} \
               Shuffling Data: {cfg.DATA.SHUFFLE} Training Samples: {len(train_dataloader.dataset)} \
               Valid Samples: {len(valid_dataloader.dataset)}"
    LOGGER.info(exp_info)
    train_utils.config_logging(os.path.join(exp_path, 'logging.txt'), cfg, access_mode='w+')

    optimizer = train_utils.create_optimizer_temp(cfg.OPTIMIZER, model)

    # Criterion (Loss function)
    def criterion_wrap(outputs, labels):
        criterion = train_utils.create_criterion(cfg.TRAIN.LOSS, n_class=cfg.MODEL.NUM_CLASSES)
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            # loss = criterion(outputs, torch.argmax(labels.long(), axis=1))
            # loss = criterion(outputs, labels[:,0])
            # TODO: long?
            loss = criterion(outputs, labels[:,0].long())
        else:
            loss = criterion(outputs, labels)
        return loss

    # Final activation
    activation_func = train_utils.create_activation(cfg.MODEL.ACTIVATION)

    # LR scheduler
    lr_scheduler = train_utils.create_lr_scheduler(
        optimizer, step_size=cfg.TRAIN.LR_SCHEDULER.decay_step, gamma=cfg.TRAIN.LR_SCHEDULER.gamma)

    trainer_instance = trainer.Trainer(
                                    model, 
                                    criterion_wrap, 
                                    optimizer, 
                                    train_dataloader, 
                                    valid_dataloader,
                                    LOGGER,
                                    device=cfg.device,
                                    n_class=cfg.MODEL.NUM_CLASSES,
                                    exp_path=exp_path,
                                    lr_scheduler=lr_scheduler,
                                    train_epoch=cfg.TRAIN.EPOCH,
                                    valid_activation=activation_func,
                                    USE_TENSORBOARD=True,
                                    checkpoint_saving_steps=cfg.TRAIN.CHECKPOINT_SAVING_STEPS,
                                    # USE_CUDA=True,)
                                    history=None)

    trainer_instance.fit()


if __name__ == '__main__':
    main(CONFIG_PATH)
    # f = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-preprocess/1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405_bboxes.npy'
    # print(np.sum(np.load(f)))
    # print(np.sum(np.load(f.replace('bboxes', 'ebox_origin'))))
    # print(np.sum(np.load(f.replace('bboxes', 'origin'))))
    # print(np.sum(np.load(f.replace('bboxes', 'spacing'))))

    # import nrrd
    # print(np.sum(nrrd.read(f.replace('bboxes.npy', 'clean.nrrd'))[0]))
    # print(np.sum(nrrd.read(f.replace('bboxes.npy', 'mask.nrrd'))[0]))
    # # print(np.sum(np.load(f.replace('bboxes', 'spacing'))))
    