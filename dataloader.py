from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import ndimage
import random

from luna16_data_preprocess import LUNA16_CropRange_Builder
from volume_generator import luna16_volume_generator
from modules.data import dataset_utils



class Luna16CropDataset(Dataset):
    def __init__(self, data_path, crop_range, mode='train'):
        self.data_path = data_path
        self.crop_range = crop_range
        self.file_name_key = LUNA16_CropRange_Builder.get_filename_key(self.crop_range)

        if mode == 'train':
            self.subsets = [f'subset{i}' for i in range(7)]
        elif mode == 'valid':
            self.subsets = ['subset7']
        elif mode == 'test':
            self.subsets = ['subset8', 'subset9']
        
        input_files = pd.read_csv(os.path.join(self.data_path, self.file_name_key, 'data_samples.csv'))
        self.input_files = input_files[input_files['subset'].isin(self.subsets)]
        
    def __len__(self):
        return self.input_files.shape[0]
    
    def __getitem__(self, idx):
        volume_data_path = self.input_files['path'].iloc[idx]
        
        raw_chunk = np.load(os.path.join(self.data_path, self.file_name_key, volume_data_path))
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