import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from dataset_conversion.crop_data_utils import get_filename_key
from nodule_classification.data.data_loader import BaseNoduleClsDataset, BaseMalignancyClsDataset, ASUSCropDataset, Luna16CropDataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


def build_coco_path(coco_root, num_fold, assign_fold=None, mode='train'):
    assert os.path.isdir(os.path.join(coco_root, f'cv-{num_fold}')), 'cv root not exist'
    if assign_fold is not None:
        assert assign_fold < num_fold, f'assign fold not exist in cv-{num_fold}'
        fold_list = [assign_fold]
    else:
        fold_list = list(range(num_fold))

    total_coco = []
    for fold in fold_list:
        coco_path = os.path.join(coco_root, f'cv-{num_fold}', str(fold))
        if mode == 'train':
            train_coco = os.path.join(coco_path, 'annotations_train.json')
            valid_coco = os.path.join(coco_path, 'annotations_test.json')
            total_coco.append((train_coco, valid_coco))
        elif mode in ['test', 'eval']:
            valid_coco = os.path.join(coco_path, 'annotations_test.json')
            total_coco.append(valid_coco)

    return total_coco


def build_dataset(data_path, crop_range, train_seriesuid, valid_seriesuid, transform, batch_size):
    train_dataset = BaseMalignancyClsDataset(
        data_path, crop_range, train_seriesuid, cls_balance=True, data_augmentation=transform)
    valid_dataset = BaseMalignancyClsDataset(
        data_path, crop_range, valid_seriesuid, cls_balance=True, data_augmentation=False)

    # train_dataset = BaseNoduleClsDataset(
    #     data_path, crop_range, train_seriesuid, cls_balance=False, data_augmentation=transform)
    # valid_dataset = BaseNoduleClsDataset(
    #     data_path, crop_range, valid_seriesuid, data_augmentation=False)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0, 
        # worker_init_fn=seed_worker, generator=g
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0, 
        # worker_init_fn=seed_worker, generator=g
    )
    return train_dataloader, valid_dataloader


    # train_datasets, valid_datasets = [], []
    # for dataset_name in config.DATA.NAME:
    #     if dataset_name == 'LUNA16':
    #         file_name_key = get_filename_key(crop_range, negative_positive_ratio)
    #         data_path = os.path.join(config.DATA.DATA_PATH[dataset_name], file_name_key)
    #         train_dataset = Luna16CropDataset(data_path, crop_range, mode='train')
    #         valid_dataset = Luna16CropDataset(data_path, crop_range, mode='valid')
    #     elif dataset_name in ['ASUS-B', 'ASUS-M']:
    #         file_name_key = get_filename_key(crop_range, negative_positive_ratio)
    #         data_path = os.path.join(config.DATA.DATA_PATH[dataset_name], file_name_key)
    #         train_dataset = ASUSCropDataset(data_path, crop_range, negative_to_positive_ratio=config.DATA.NPratio_test, 
    #                                         nodule_type=dataset_name, mode='train', data_augmentation=config.DATA.IS_DATA_AUGMENTATION)
    #         valid_dataset = ASUSCropDataset(data_path, crop_range, negative_to_positive_ratio=config.DATA.NPratio_test, 
    #                                         nodule_type=dataset_name, mode='valid', data_augmentation=config.DATA.IS_DATA_AUGMENTATION)
    #     print('Train number', len(train_dataset), 'Valid number', len(valid_dataset))
    #     train_datasets.append(train_dataset)
    #     valid_datasets.append(valid_dataset)

    # total_train_dataset, total_valid_dataset = torch.utils.data.ConcatDataset(train_datasets), torch.utils.data.ConcatDataset(valid_datasets)
    # train_dataloader = DataLoader(total_train_dataset, batch_size=config.DATA.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
    # valid_dataloader = DataLoader(total_valid_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)
    # return train_dataloader, valid_dataloader