# import os
# import torch
from torch.utils.data import Dataset, DataLoader

from dataset_conversion.crop_data_utils import get_filename_key
from nodule_classification.data.data_loader import BaseCropClsDataset, ASUSCropDataset, Luna16CropDataset


def build_dataset(data_path, crop_range, train_seriesuid, valid_seriesuid, transform, batch_size):
    train_dataset = BaseCropClsDataset(data_path, crop_range, train_seriesuid, transform)
    valid_dataset = BaseCropClsDataset(data_path, crop_range, valid_seriesuid, data_augmentation=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)
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