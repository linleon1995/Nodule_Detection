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
from nodule_classification.model.ResNet_3d import build_3d_resnet

from nodule_classification.model.MatchingNet import MatchingNetwork_3d
from nodule_classification.data.matchnet_utils import build_support_set, matchingnet_trainer

# import torch.utils.tensorboard
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
    for (fold, train_coco, valid_coco) in coco_list:
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

    if cfg.MODEL.NAME == '3dResnet':
        model = build_3d_resnet(
            model_depth=cfg.MODEL.DEPTH, n_classes=cfg.MODEL.NUM_CLASSES, conv1_t_size=7, conv1_t_stride=2)
    elif cfg.MODEL.NAME == 'MatchingNet':
        model = MatchingNetwork_3d(keep_prob=0.9, 
                                   batch_size=cfg.DATA.BATCH_SIZE,
                                   num_channels=3, 
                                   learning_rate=cfg.OPTIMIZER.learning_rate, 
                                   fce=False,
                                   use_cuda=True,
                                   model_depth=cfg.MODEL.DEPTH,
                                   n_classes=cfg.MODEL.NUM_CLASSES,)

    # Build dataset
    train_seriesuid = get_pids_from_coco(train_coco)
    valid_seriesuid = get_pids_from_coco(valid_coco)
    train_dataloader, valid_dataloader = build_dataset(
        cfg.DATA.DATA_PATH[cfg.DATA.NAME], cfg.DATA.CROP_RANGE, train_seriesuid, valid_seriesuid,
        cfg.DATA.IS_DATA_AUGMENTATION, cfg.DATA.BATCH_SIZE, task=cfg.DATA.TASK)
    if cfg.MODEL.NAME == 'MatchingNet':
        support_set_x, support_set_y = build_support_set(n=64, n_class=cfg.MODEL.NUM_CLASSES)

    # Logger
    LOGGER.info("Start Training!!")
    LOGGER.info(f"Training epoch: {cfg.TRAIN.EPOCH} Batch size: {cfg.DATA.BATCH_SIZE}")
    LOGGER.info(f"Shuffling Data: {cfg.DATA.SHUFFLE} Training Samples: {len(train_dataloader.dataset)}")
    LOGGER.info(f"Valid Samples: {len(valid_dataloader.dataset)}")
    train_utils.config_logging(os.path.join(exp_path, 'logging.txt'), cfg, access_mode='w+')

    # TODO: the lr is hided, not a good representation
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

    if cfg.MODEL.NAME == 'MatchingNet':
        model_trainer = matchingnet_trainer
        trainer_instance = model_trainer(
                                        model, 
                                        criterion_wrap, 
                                        optimizer, 
                                        train_dataloader, 
                                        valid_dataloader,
                                        LOGGER,
                                        support_set_x=support_set_x,
                                        support_set_y=support_set_y,
                                        device=cfg.device,
                                        n_class=cfg.MODEL.NUM_CLASSES,
                                        exp_path=exp_path,
                                        lr_scheduler=lr_scheduler,
                                        train_epoch=cfg.TRAIN.EPOCH,
                                        valid_activation=activation_func,
                                        USE_TENSORBOARD=True,
                                        checkpoint_saving_steps=cfg.TRAIN.CHECKPOINT_SAVING_STEPS,
                                        # USE_CUDA=True,)
                                        history=cfg.TRAIN.INIT_CHECKPOINT,
                                        patience=cfg.TRAIN.PATIENCE,
                                        )
    else:
        model_trainer = trainer.Trainer
        trainer_instance = model_trainer(
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
                                        history=cfg.TRAIN.INIT_CHECKPOINT,
                                        patience=cfg.TRAIN.PATIENCE)

    trainer_instance.fit()


if __name__ == '__main__':
    main(CONFIG_PATH)

    # from data.data_utils import get_files, load_itk
    # import matplotlib.pyplot as plt
    # f = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule\TMH-Benign\1B0036\1B0036mask mhd\1.2.826.0.1.3680043.2.1125.1.72517941620469000392115845233221023.mhd'
    # f2 = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule\TMH-Benign\1B012\1B012 mask mhd\1.2.826.0.1.3680043.2.1125.1.37191595439617774847726210923622088.mhd'
    # x_f = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule\TMH-Benign\1B0036\1B0036 raw mhd\1.2.826.0.1.3680043.2.1125.1.38467158349469692405660363178115017.mhd'
    # x_f2 = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule\TMH-Benign\1B012\1B012 raw mhd\1.2.826.0.1.3680043.2.1125.1.80440926958465866612590512270527764.mhd'
    # y, _, _, _ = load_itk(f)
    # y2, _, _, _ = load_itk(f2)
    # x, _, _, _ = load_itk(x_f)
    # x2, _, _, _ = load_itk(x_f2)

    # y2 = y
    # x2 = x
    # for i in range(x.shape[0]):
    #     print(i)
    #     if np.sum(y[i]):
    #         plt.title(f'slice {i}')
    #         plt.imshow(x[i], 'gray')
    #         plt.imshow(y[i], alpha=0.2)
    #         plt.show()
    
    