import site_path
import os
from ResNet_3d import build_3d_resnet
from torch.utils.data import Dataset, DataLoader
from dataloader import Luna16CropDataset
import numpy as np
import random
import torch
from pprint import pprint
import tensorboardX

from modules.train import trainer
from modules.utils import configuration
from modules.utils import train_utils

CONFIG_PATH = 'config/train_config.yml'
LOGGER = train_utils.get_logger('train')


def main(config_reference):
    # Configuration
    config = configuration.load_config(config_reference)
    config = train_utils.DictAsMember(config)
    device = configuration.get_device(config.get('device', None))
    config.device = device

    checkpoint_root = os.path.join(config.TRAIN.PROJECT_PATH, 'checkpoints')
    checkpoint_path = train_utils.create_training_path(checkpoint_root)
    config.CHECKPOINT_PATH = checkpoint_path
    pprint(config)

    # Set deterministic
    manual_seed = config.TRAIN.manual_seed
    if manual_seed is not None:
        LOGGER.info(f'Seed the RNG for all devices with {manual_seed}')
        train_utils.set_deterministic(manual_seed, random, np, torch)

    model = build_3d_resnet(model_depth=config.MODEL.DEPTH, n_classes=config.MODEL.NUM_CLASSES, conv1_t_size=7, conv1_t_stride=2)
    print(model)

    train_dataset = Luna16CropDataset(config.DATA.DATA_PATH, config.DATA.CROP_RANGE, mode='train')
    valid_dataset = Luna16CropDataset(config.DATA.DATA_PATH, config.DATA.CROP_RANGE, mode='valid')

    train_dataloader = DataLoader(train_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    # Logger
    LOGGER.info("Start Training!!")
    LOGGER.info("Training epoch: {} Batch size: {} Shuffling Data: {} Training Samples: {}".
            format(config.TRAIN.EPOCH, config.DATA.BATCH_SIZE, config.DATA.SHUFFLE, len(train_dataloader.dataset)))
    train_utils.config_logging(os.path.join(checkpoint_path, 'logging.txt'), config, access_mode='w+')

    optimizer = train_utils.create_optimizer_temp(config.OPTIMIZER, model)

    # Criterion (Loss function)
    def criterion_wrap(outputs, labels):
        criterion = train_utils.create_criterion(config.TRAIN.LOSS)
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            # loss = criterion(outputs, torch.argmax(labels.long(), axis=1))
            loss = criterion(outputs, labels[:,0])
        else:
            loss = criterion(outputs, labels)
        return loss

    # Final activation
    activation_func = train_utils.create_activation(config.MODEL.ACTIVATION)

    # TODO: device change to captial
    trainer_instance = trainer.Trainer(config,
                                    model, 
                                    criterion_wrap, 
                                    optimizer, 
                                    train_dataloader, 
                                    valid_dataloader,
                                    logger=LOGGER,
                                    device=config.device,
                                    activation_func=activation_func,
                                    USE_TENSORBOARD=True,
                                    USE_CUDA=True)

    trainer_instance.fit()


if __name__ == '__main__':
    main(CONFIG_PATH)