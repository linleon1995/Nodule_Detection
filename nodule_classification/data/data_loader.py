from torch import positive
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import ndimage
import random
from data.volume_generator import luna16_volume_generator
from nodule_classification.data.tranaform_3d import image_3d_cls_transform




class BaseNoduleClsDataset(Dataset):
    def __init__(self, data_path, crop_range, seriesuid, cls_balance=True, data_augmentation=False):
        self.data_path = data_path
        self.crop_range = crop_range
        self.seriesuid = seriesuid
        self.data_augmentation = data_augmentation

        input_files = pd.read_csv(os.path.join(self.data_path, 'data_samples.csv'))
        input_files = input_files[input_files['seriesuid'].isin(self.seriesuid)]
        if cls_balance:
            input_files = self.class_balance(input_files)
        self.input_files = input_files

    def __len__(self):
        return self.input_files.shape[0]
    
    def __getitem__(self, idx):
        row_df = self.input_files.iloc[idx]
        volume_data_path = row_df['path']

        # volume_data_path = self.input_files['path'].iloc[idx]
        
        raw_chunk = np.load(os.path.join(self.data_path, volume_data_path))
        if self.data_augmentation:
            raw_chunk = self.transform(raw_chunk)
        raw_chunk = np.float32(np.tile(raw_chunk[np.newaxis], (3,1,1,1)))
        # target = np.array([0, 1]) if row_df['category'] == 'positive' else np.array([1, 0])
        target = 1 if row_df['category'] == 'positive' else 0
        target = np.array(target, dtype='float')[np.newaxis]
        return {'input':raw_chunk, 'target': target}

    def transform(self, img):
        img, _ = image_3d_cls_transform(img)
        return img

    def class_balance(self, input_files):
        positive_pd = input_files[input_files.category == 'positive']
        negative_pd = input_files[input_files.category == 'negative']
        negative_pd = negative_pd.sample(n=positive_pd.shape[0], random_state=1)
        input_files = pd.concat([positive_pd, negative_pd])
        return input_files

    # def class_balance(self, input_files):
    #     positive_pd = input_files[input_files.category == 'positive']
    #     negative_pd = input_files[input_files.category == 'negative']
    #     pid_seq = input_files.seriesuid.unique()
    #     negative_list = []
    #     for pid in pid_seq:
    #         row_df = negative_pd[negative_pd.seriesuid == pid]
    #         # TODO: why no negative exist in some cases?
    #         if row_df.shape[0] > 0:
    #             row_df = row_df.sample(n=1)
    #             negative_list.append(row_df)
    #     total_pd = [positive_pd] + negative_list
    #     input_files = pd.concat(total_pd)
    #     return input_files


class BaseMalignancyClsDataset(BaseNoduleClsDataset):
    def __init__(self, data_path, crop_range, seriesuid, cls_balance=True, data_augmentation=False):
        super().__init__(data_path, crop_range, seriesuid, cls_balance, data_augmentation)
    
    def __getitem__(self, idx):
        row_df = self.input_files.iloc[idx]
        volume_data_path = row_df['path']
        
        raw_chunk = np.load(os.path.join(self.data_path, volume_data_path))
        if self.data_augmentation:
            raw_chunk = self.transform(raw_chunk)
        raw_chunk = np.float32(np.tile(raw_chunk[np.newaxis], (3,1,1,1)))

        # if isinstance(row_df['malignancy'], float):
        #     if np.isnan(row_df['malignancy']):
        #         target = 0
        if False:
            pass
        elif row_df['malignancy'] == 'benign':
            target = 1
        elif row_df['malignancy'] == 'malignant':
            target = 2
        else:
            target = 0

        target = np.array(target, dtype='float')[np.newaxis]
        return {'input':raw_chunk, 'target': target}

    def class_balance(self, input_files):
        benign_pd = input_files[input_files.malignancy == 'benign']
        malignancy_pd = input_files[input_files.malignancy == 'malignant']
        negative_pd = input_files[input_files.malignancy.isnull()]
        positive_pd = pd.concat([benign_pd, malignancy_pd])
        negative_pd = negative_pd.sample(n=positive_pd.shape[0], random_state=1)
        input_files = pd.concat([positive_pd, negative_pd])
        return input_files


class BaseNoduleClsDataset(Dataset):
    def __init__(self, data_path, crop_range, seriesuid, cls_balance=True, data_augmentation=False):
        self.data_path = data_path
        self.crop_range = crop_range
        self.seriesuid = seriesuid
        self.data_augmentation = data_augmentation

        input_files = pd.read_csv(os.path.join(self.data_path, 'data_samples.csv'))
        input_files = input_files[input_files['seriesuid'].isin(self.seriesuid)]
        if cls_balance:
            input_files = self.class_balance(input_files)
        self.input_files = input_files

    def __len__(self):
        return self.input_files.shape[0]
    
    def __getitem__(self, idx):
        row_df = self.input_files.iloc[idx]
        volume_data_path = row_df['path']

        # volume_data_path = self.input_files['path'].iloc[idx]
        
        raw_chunk = np.load(os.path.join(self.data_path, volume_data_path))
        if self.data_augmentation:
            raw_chunk = self.transform(raw_chunk)
        raw_chunk = np.float32(np.tile(raw_chunk[np.newaxis], (3,1,1,1)))
        # target = np.array([0, 1]) if row_df['category'] == 'positive' else np.array([1, 0])
        target = 1 if row_df['category'] == 'positive' else 0
        target = np.array(target, dtype='float')[np.newaxis]
        malignancy = row_df['malignancy']
        return {'input':raw_chunk, 'target': target, 'malignancy': malignancy}

    def transform(self, img):
        img, _ = image_3d_cls_transform(img)
        return img

    def class_balance(self, input_files):
        positive_pd = input_files[input_files.category == 'positive']
        negative_pd = input_files[input_files.category == 'negative']
        negative_pd = negative_pd.sample(n=positive_pd.shape[0], random_state=1)
        input_files = pd.concat([positive_pd, negative_pd])
        return input_files

    # def class_balance(self, input_files):
    #     positive_pd = input_files[input_files.category == 'positive']
    #     negative_pd = input_files[input_files.category == 'negative']
    #     pid_seq = input_files.seriesuid.unique()
    #     negative_list = []
    #     for pid in pid_seq:
    #         row_df = negative_pd[negative_pd.seriesuid == pid]
    #         # TODO: why no negative exist in some cases?
    #         if row_df.shape[0] > 0:
    #             row_df = row_df.sample(n=1)
    #             negative_list.append(row_df)
    #     total_pd = [positive_pd] + negative_list
    #     input_files = pd.concat(total_pd)
    #     return input_files


class ASUSCropDataset(Dataset):
    def __init__(self, data_path, crop_range, nodule_type, negative_to_positive_ratio=1, mode='train', data_augmentation=False):
        self.data_path = data_path
        self.crop_range = crop_range
        self.nodule_type = nodule_type
        self.seriesuid = self.get_seriesuid(nodule_type, mode)
        self.data_augmentation = data_augmentation

        input_files = pd.read_csv(os.path.join(self.data_path, 'data_samples.csv'))
        # input_files = input_files.sample(frac=1)
        # input_files.to_csv(os.path.join(self.data_path, 'data_samples2.csv'))
        input_files = input_files[input_files['seriesuid'].isin(self.seriesuid)]
        positive_files = input_files[input_files.category == 'positive']
        negative_files = input_files[input_files.category == 'negative']
        num_positive = positive_files.shape[0]
        num_negative = int(num_positive*negative_to_positive_ratio)
        negative_subset = negative_files.iloc[:num_negative]
        self.input_files = pd.concat([positive_files, negative_subset])

    def __len__(self):
        return self.input_files.shape[0]
    
    def __getitem__(self, idx):
        volume_data_path = self.input_files['path'].iloc[idx]
        
        raw_chunk = np.load(os.path.join(self.data_path, volume_data_path))
        if self.data_augmentation:
            raw_chunk = self.transform(raw_chunk)
        raw_chunk = np.float32(np.tile(raw_chunk[np.newaxis], (3,1,1,1)))
        target = 1 if 'positive' in volume_data_path else 0
        target = np.array(target, dtype='float')[np.newaxis]
        return {'input':raw_chunk, 'target': target}

    # def get_seriesuid(self, nodule_type, mode):
    #     if nodule_type == 'ASUS-B':
    #         if mode == 'train':
    #             seriesuid = [f'1B{i:03d}' for i in range(11, 36)]
    #         elif mode == 'valid':
    #             seriesuid = [f'1B{i:03d}' for i in range(9, 11)]
    #         elif mode == 'test':
    #             seriesuid = [f'1B{i:03d}' for i in range(9)]
    #     elif nodule_type == 'ASUS-M':
    #         if mode == 'train':
    #             seriesuid = [f'1m{i:04d}' for i in range(18, 58)]
    #         elif mode == 'valid':
    #             seriesuid = [f'1m{i:04d}' for i in range(13, 18)]
    #         elif mode == 'test':
    #             seriesuid = [f'1m{i:04d}' for i in range(18)]
    #     return seriesuid

    def get_seriesuid(self, nodule_type, mode):
        if nodule_type == 'ASUS-B':
            if mode == 'train':
                seriesuid = [f'1B{i:03d}' for i in range(1, 18)]
            elif mode == 'valid':
                seriesuid = [f'1B{i:03d}' for i in range(18, 20)]
            elif mode == 'test':
                seriesuid = [f'1B{i:03d}' for i in range(20, 26)]
        elif nodule_type == 'ASUS-M':
            if mode == 'train':
                seriesuid = [f'1m{i:04d}' for i in range(1, 35)]
            elif mode == 'valid':
                seriesuid = [f'1m{i:04d}' for i in range(35, 37)]
            elif mode == 'test':
                seriesuid = [f'1m{i:04d}' for i in range(37, 45)]
        return seriesuid

    def transform(self, volume):
        volume = self.random_flip_3d(volume)
        return volume

    def random_flip_3d(self, volume):
        random_prob = np.random.random(3)
        for axis, p in enumerate(random_prob):
            if p > 0.5:
                volume = np.flip(volume, axis=axis)
        return volume

    # TODO: check the behavior is correct
    def rotate(volume):
        """Rotate the volume by a few degrees"""

        def scipy_rotate(volume):
            # define some rotation angles
            angles = [-15, -10, -5, 5, 10, 15]
            # pick angles at random
            angle = random.choice(angles)
            # rotate volume
            volume = ndimage.rotate(volume, angle, reshape=False)
            return volume

        augmented_volume = scipy_rotate(volume)
        return augmented_volume


class Luna16CropDataset(Dataset):
    def __init__(self, data_path, crop_range, mode='train'):
        self.data_path = data_path
        self.crop_range = crop_range

        if mode == 'train':
            self.subsets = [f'subset{i}' for i in range(7)]
        elif mode == 'valid':
            self.subsets = ['subset7']
        elif mode == 'test':
            self.subsets = ['subset8', 'subset9']
        
        input_files = pd.read_csv(os.path.join(self.data_path, 'data_samples.csv'))
        self.input_files = input_files[input_files['subset'].isin(self.subsets)]
        
    def __len__(self):
        return self.input_files.shape[0]
    
    def __getitem__(self, idx):
        volume_data_path = self.input_files['path'].iloc[idx]
        
        raw_chunk = np.load(os.path.join(self.data_path, volume_data_path))
        raw_chunk = np.float32(np.tile(raw_chunk[np.newaxis], (3,1,1,1)))
        target = 1 if 'positive' in volume_data_path else 0
        target = np.array(target, dtype='float')[np.newaxis]
        return {'input':raw_chunk, 'target': target}

    def random_flip_3d(self, volume):
        random_prob = np.random.random(3)
        for axis, p in enumerate(random_prob):
            if p > 0.5:
                volume = np.flip(volume, axis=axis)
        return volume

    # TODO: check the behavior is correct
    def rotate(volume):
        """Rotate the volume by a few degrees"""

        def scipy_rotate(volume):
            # define some rotation angles
            angles = [-15, -10, -5, 5, 10, 15]
            # pick angles at random
            angle = random.choice(angles)
            # rotate volume
            volume = ndimage.rotate(volume, angle, reshape=False)
            return volume

        augmented_volume = scipy_rotate(volume)
        return augmented_volume