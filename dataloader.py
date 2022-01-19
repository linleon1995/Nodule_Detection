from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import time
from luna16_data_preprocess import LUNA16_CropRange_Builder
from volume_generator import luna16_volume_generator
from modules.data import dataset_utils


class Luna16CropDataset(Dataset):
    def __init__(self, data_path, crop_range):
        self.data_path = data_path
        self.crop_range = crop_range
        file_name_key = LUNA16_CropRange_Builder.get_filename_key(self.crop_range)
        self.input_files = dataset_utils.get_files(os.path.join(self.data_path, file_name_key, 'Image'), 'npy')
        self.target_files = dataset_utils.get_files(os.path.join(self.data_path, file_name_key, 'Mask'), 'npy')
        assert len(self.input_files) == len(self.target_files), 'Mismatch file number'
        
    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        raw_chunk = np.load(self.input_files[idx])
        target_chunk = np.load(self.target_files[idx])
        return {'input':raw_chunk, 'target': 1}


if __name__ == '__main__':
    crop_range = {'index': 64, 'row': 64, 'column': 64}
    crop_range_path = 'LUNA16_infos'
    dataset = Luna16CropDataset(crop_range, crop_range_path)
    train_dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
    # total_time = 0

    # start_time = time.time()
    for i, data in enumerate(train_dataloader):
        # end_time = time.time()
        print(i, data['input'].shape)
        # total_time +=end_time-start_time
        # start_time = time.time()
    # print(total_time)