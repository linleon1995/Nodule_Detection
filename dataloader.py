from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import time
from luna16_data_preprocess import LUNA16_CropRange_Builder
from volume_generator import luna16_volume_generator
from modules.data import dataset_utils


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
        raw_chunk = np.load(data_path)
        raw_chunk = np.float32(np.tile(raw_chunk[np.newaxis], (3,1,1,1)))
        target = 1 if 'positive' in data_path else 0
        target = np.array(target, dtype='float')[np.newaxis]
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