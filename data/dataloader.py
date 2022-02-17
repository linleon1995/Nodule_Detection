from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import ndimage
import random
from utils.volume_generator import luna16_volume_generator
from modules.data import dataset_utils


# TODO: simple factory


class ASUSCropDataset(Dataset):
    def __init__(self, data_path, crop_range, nodule_type, negative_to_positive_ratio=1, mode='train', data_augmentation=False):
        self.data_path = data_path
        self.crop_range = crop_range
        self.nodule_type = nodule_type
        self.seriesuid = self.get_seriesuid(nodule_type, mode)
        self.data_augmentation = data_augmentation

        input_files = pd.read_csv(os.path.join(self.data_path, 'data_samples.csv'))
        input_files = input_files[input_files['seriesuid'].isin(self.seriesuid)]
        positive_files = input_files[input_files.category == 'positive']
        negative_files = input_files[input_files.category == 'negative']
        num_positive = positive_files.shape[0]
        num_negative = int(num_positive*negative_to_positive_ratio)
        self.input_files = pd.concat([positive_files, negative_files.iloc[:num_negative]])

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

    def get_seriesuid(self, nodule_type, mode):
        if nodule_type == 'ASUS-B':
            if mode == 'train':
                seriesuid = [f'1B{i:03d}' for i in range(0, 25)]
            elif mode == 'valid':
                seriesuid = [f'1B{i:03d}' for i in range(25, 27)]
            elif mode == 'test':
                seriesuid = [f'1B{i:03d}' for i in range(27, 35)]
        elif nodule_type == 'ASUS-M':
            if mode == 'train':
                seriesuid = [f'1m{i:04d}' for i in range(0, 40)]
            elif mode == 'valid':
                seriesuid = [f'1m{i:04d}' for i in range(40, 45)]
            elif mode == 'test':
                seriesuid = [f'1m{i:04d}' for i in range(45, 57)]
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