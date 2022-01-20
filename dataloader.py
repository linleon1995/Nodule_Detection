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
        file_name_key = LUNA16_CropRange_Builder.get_filename_key(self.crop_range)
        if mode == 'train':
            self.subsets = [f'subset{i}' for i in range(7)]
        elif mode == 'valid':
            self.subsets = ['subset7']
        elif mode == 'test':
            self.subsets = ['subset8', 'subset9']
        
        input_files = pd.read_csv(os.path.join(self.data_path, file_name_key, 'data_samples.csv'))
        self.input_files = input_files[input_files['subset'].isin(self.subsets)]
        
    def __len__(self):
        return self.input_files.shape[0]

    def __getitem__(self, idx):
        data_path = self.input_files['path'].iloc[idx]
        
        # data_path = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\crop\backup\64x64x64\positive\Image'
        # # data_path = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\crop\backup\64x64x64\negative\Image'
        # data_path = os.path.join(data_path, rf'luna16-0003-1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.npy')

        raw_chunk = np.load(data_path)
        # raw_chunk = rotate(raw_chunk)
        raw_chunk = np.float32(np.tile(raw_chunk[np.newaxis], (3,1,1,1)))

        target = 1 if 'positive' in data_path else 0
        target = np.array(target, dtype='float')[np.newaxis]

        mask_path = data_path.replace('Image', 'Mask')
        mask_chunk = np.load(mask_path)
        # mask_chunk = rotate(mask_chunk)
        for i in range(0, 64, 30):
            if np.sum(mask_chunk[i]) == 0:
                plt.imshow(raw_chunk[0,i], 'gray')
                plt.imshow(mask_chunk[i], alpha=0.2)
                plt.title(f'{i}-{target}')
                plt.show()

        return {'input':raw_chunk, 'target': target}


if __name__ == '__main__':
    crop_range = {'index': 64, 'row': 64, 'column': 64}
    crop_range_path = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\crop'
    train_dataset = Luna16CropDataset(crop_range_path, crop_range, mode='train')
    valid_dataset = Luna16CropDataset(crop_range_path, crop_range, mode='valid')

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)

    for i, data in enumerate(train_dataloader):
        print(i, data['input'].shape, data['target'])

    for i, data in enumerate(valid_dataloader):
        print(i, data['input'].shape, data['target'])