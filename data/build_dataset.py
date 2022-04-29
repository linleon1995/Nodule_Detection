
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd

from data.volume_generator import luna16_volume_generator
from data.data_utils import get_files, load_itk
from utils.utils import mask_preprocess, raw_preprocess
from nodule_classification.data.luna16_crop_preprocess import LUNA16_CropRange_Builder
# from data.data_utils import get_files
# from data.data_transformer import ImageDataTransformer


class MhdNoduleDataset():
    def __init__(self, pid_list, input_path_list, target_path_list, crop_range, annotation_df, mode,
                 data_transformer=None, nb_samples=1):
        self.pid_list = pid_list
        self.input_path_list = input_path_list
        self.target_path_list = target_path_list
        self.crop_range = crop_range
        self.crop_range_dict = {'index': crop_range[0], 'row': crop_range[1], 'column': crop_range[2]}
        self.data_transformer = data_transformer
        self.nb_samples = nb_samples
        self.get_idx = lambda length, crop_length: np.random.randint(0, length-crop_length+1)
        self.annotation_df = annotation_df
        self.mode = mode
        
        # TODO:
        self.random_shift_range = 0.2

    def __len__(self):
        return len(self.input_path_list)

    def __getitem__(self, idx):
        input_data, target = [], []
        
        pid = os.path.split(self.input_path_list[idx])[1][:-4]
        df = self.annotation_df.loc[self.annotation_df['seriesuid'] == pid]
        vol, origin_zyx, spacing_zyx, direction_zyx = load_itk(self.input_path_list[idx])
        vol = np.clip(vol, -1000, 1000)
        vol = raw_preprocess(vol, output_dtype=np.float32)
        vol = vol[...,0] / 255

        mask_vol = np.load(self.target_path_list[pid])
        # mask_vol = mask_preprocess(mask_vol)

        
        # Get positive sample
        if df.shape[0] == 0:
            nb_samples = self.nb_samples * 2 # If no nodule exist then load double negative
        else:
            nb_samples = self.nb_samples
            df = df.sample(nb_samples)
        
        for index, data_info in df.iterrows():
            center_zyx = [data_info['coordZ'], data_info['coordY'], data_info['coordX']]
            center_cri = LUNA16_CropRange_Builder.xyz2irc(center_zyx, origin_zyx, spacing_zyx, direction_zyx)

            if self.mode =='train':
                if self.random_shift_range:
                    rand_num = (np.random.random(3)*2 - 1) * self.random_shift_range
                    crop_range_arr = np.array(self.crop_range)
                    rand_shift = np.int32(rand_num*self.crop_range)
                    center_cri = center_cri + rand_shift
                    center_cri = np.clip(center_cri, crop_range_arr//2, vol.shape[::-1]-crop_range_arr//2)

            # center = {'index': center_cri[0], 'row': center_cri[1], 'column': center_cri[2]}
            center = {'index': center_cri[2], 'row': center_cri[1], 'column': center_cri[0]}
            input_crop = LUNA16_CropRange_Builder.crop_volume(vol, self.crop_range_dict, center)
            mask_crop = LUNA16_CropRange_Builder.crop_volume(mask_vol, self.crop_range_dict, center)

            # from utils.vis import show_mask_base
            # dir_name = str(np.random.random())
            # print(dir_name, center_cri)
            # # print(dir_name, center_cri, rand_num, rand_shift)
            # save_dir = os.path.join('plot', '3d_SHIFT', dir_name)
            # os.makedirs(save_dir, exist_ok=True)
            # for i in range(32):
            #     if np.sum(mask_crop[i]):
            #         show_mask_base(input_crop[i], mask_crop[i], save_path=os.path.join(save_dir, f'img_{i}.png'))

 
            input_data.append(input_crop)
            target.append(mask_crop)

            # from utils.vis import show_mask_base
            # for ii in range(input_crop.shape[0]):
            #     if np.sum(mask_crop[ii])> 0:
            #         show_mask_base(input_crop[ii], mask_crop[ii], save_path=f'{ii}.png')

        
        # Get negative sample
        num = 0
        while True:
            slice_idx = []
            for idx, length in enumerate(list(vol.shape)):
                start_idx = self.get_idx(length, self.crop_range[idx])
                slice_idx.append(slice(start_idx, start_idx+self.crop_range[idx]))

            mask_crop = mask_vol[slice_idx[0], slice_idx[1], slice_idx[2]]
            if np.sum(mask_crop) > 0:
                continue
            else:
                input_crop = vol[slice_idx[0], slice_idx[1], slice_idx[2]]
                input_data.append(input_crop)
                target.append(mask_crop)
                num += 1

            if num >= nb_samples:
                break

        input_data = np.stack(input_data)
        target = np.stack(target)
        
        input_data = np.transpose(input_data, (0, 2, 3, 1))
        target = np.transpose(target, (0, 2, 3, 1))
        return input_data, target


class GeneralDataset():
    def __init__(self, input_path_list, target_path_list, input_load_func, target_load_func, data_transformer=None):
        self.input_load_func = input_load_func
        self.target_load_func = target_load_func
        self.input_path_list = input_path_list
        self.target_path_list = target_path_list
        self.data_transformer = data_transformer
    
    def __len__(self):
        return len(self.input_path_list)

    def __getitem__(self, idx):
        input_data, target = self.input_load_func(self.input_path_list[idx]), self.target_load_func(self.target_path_list[idx])
        input_data, target = np.swapaxes(np.swapaxes(input_data, 0, 1), 1, 2), np.swapaxes(np.swapaxes(target, 0, 1), 1, 2)
        if self.data_transformer is not None:
            input_data, target = self.data_transformer(input_data, target)
        input_data = input_data / 255
        input_data, target = input_data[np.newaxis], target[np.newaxis]

        # TODO: related issue: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/12
        # Re-assign array memory beacause of the flipping operation
        input_data, target = input_data.copy(), target.copy()
        return input_data, target


def flip(x, y):
    dim = len(x.shape)
    for axis in range(dim):
    # for axis in range(1,2):
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=axis)
            y = np.flip(y, axis=axis)
    return x, y



def build_dataloader_mhd(input_roots, target_roots, train_cases, annot_path, valid_cases=None, train_batch_size=1, pin_memory=True, 
                     num_workers=0, transform_config=None, class_balance=False, remove_empty_sample=True, 
                     crop_range=(32,64,64)):

    transformer = None
    assert (not remove_empty_sample or not class_balance), 'Remove empty sample might conflict with class balncing'
    annotation_df = pd.read_csv(annot_path)

    def get_target_samples(sub_input_paths, target_paths):
        target_samples = {}
        for input_path in sub_input_paths:
            filename = os.path.split(input_path)[1][:-4]
            for target_path in target_paths:
                if filename in target_path:
                    target_samples[filename] = target_path
                    break
        return target_samples

    target_samples = []
    train_input_samples = []
    train_pid_list = []
    for input_root, target_root in zip(input_roots, target_roots):
        all_train_input_paths = get_files(input_root, keys=train_cases, get_dirs=True)

        for train_input_path in all_train_input_paths:
            train_input_paths = get_files(train_input_path, keys='mhd')
            pids = get_files(train_input_path, keys='mhd', return_fullpath=False, ignore_suffix=True)
            train_input_samples.extend(train_input_paths)
            train_pid_list.extend(pids)

        target_paths = get_files(target_root, keys='npy')
        target_samples.extend(target_paths)
    train_target_samples = get_target_samples(train_input_samples, target_samples)
    
    train_dataset = MhdNoduleDataset(
        train_pid_list, train_input_samples, train_target_samples, data_transformer=transformer, crop_range=crop_range, annotation_df=annotation_df, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)

    valid_dataloader = None
    if valid_cases is not None:
        valid_input_samples = []
        valid_pid_list = []
        for input_root, target_root in zip(input_roots, target_roots):
            all_valid_input_paths = get_files(input_root, keys=valid_cases, get_dirs=True)

            for valid_input_path in all_valid_input_paths:
                valid_input_paths = get_files(valid_input_path, keys='mhd')
                pids = get_files(valid_input_path, keys='mhd', return_fullpath=False, ignore_suffix=True)
                valid_input_samples.extend(valid_input_paths)
                valid_pid_list.extend(pids)

        valid_target_samples = get_target_samples(valid_input_samples, target_paths)

        valid_dataset = MhdNoduleDataset(valid_pid_list, valid_input_samples, valid_target_samples, crop_range=crop_range, annotation_df=annotation_df, mode='valid')
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    
    return train_dataloader, valid_dataloader


def build_dataloader(input_roots, target_roots, train_cases, valid_cases=None, train_batch_size=1, pin_memory=True, 
                     num_workers=0, transform_config=None, class_balance=False, remove_empty_sample=True):
    input_load_func = target_load_func = np.load
    transformer = flip
    assert (not remove_empty_sample or not class_balance), 'Remove empty sample might conflict with class balncing'
    
    train_input_samples, train_target_samples = [], []
    for input_root, target_root in zip(input_roots, target_roots):
        all_train_input_paths = get_files(input_root, keys=train_cases, get_dirs=True)
        all_train_target_paths = get_files(target_root, keys=train_cases, get_dirs=True)

        for train_input_paths, train_target_paths in zip(all_train_input_paths, all_train_target_paths):
            train_input_paths = get_files(train_input_paths)
            train_target_paths = get_files(train_target_paths)
            train_input_samples.extend(train_input_paths)
            train_target_samples.extend(train_target_paths)

    if class_balance and not remove_empty_sample:
        p_input_path, p_target_path = [], []
        n_input_path, n_target_path = [], []
        for input_path, target_path in zip(train_input_samples, train_target_samples):
            if 'positive' in input_path:
                p_input_path.append(input_path)
                p_target_path.append(target_path)
            else:
                n_input_path.append(input_path)
                n_target_path.append(target_path)
            
        np.random.shuffle(p_input_path)
        np.random.shuffle(p_target_path)
        np.random.shuffle(n_input_path)
        np.random.shuffle(n_target_path)
        num_sample = min(len(p_input_path), len(n_input_path))
        train_input_samples = p_input_path[:num_sample] + n_input_path[:num_sample]
        train_target_samples = p_target_path[:num_sample] + n_target_path[:num_sample]

    if remove_empty_sample:
        for input_path, target_path in zip(train_input_samples, train_target_samples):
            if np.sum(np.load(target_path)) <= 0:
                train_input_samples.remove(input_path)
                train_target_samples.remove(target_path)
    
    # print(len(train_target_samples))
    train_dataset = GeneralDataset(
        train_input_samples, train_target_samples, input_load_func, target_load_func, data_transformer=transformer)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)

    valid_dataloader = None
    if valid_cases is not None:
        valid_input_samples, valid_target_samples = [], []
        for input_root, target_root in zip(input_roots, target_roots):
            all_valid_input_paths = get_files(input_root, keys=valid_cases, get_dirs=True)
            all_valid_target_paths = get_files(target_root, keys=valid_cases, get_dirs=True)

            for valid_input_paths, valid_target_paths in zip(all_valid_input_paths, all_valid_target_paths):
                valid_input_paths = get_files(valid_input_paths)
                valid_target_paths = get_files(valid_target_paths)
                valid_input_samples.extend(valid_input_paths)
                valid_target_samples.extend(valid_target_paths)  

        valid_dataset = GeneralDataset(valid_input_samples, valid_target_samples, input_load_func, target_load_func)
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    
    return train_dataloader, valid_dataloader


def build_dataloader_tmh(input_roots, target_roots, train_cases, valid_cases=None, train_batch_size=1, pin_memory=True, 
                     num_workers=0, transform_config=None, class_balance=False, remove_empty_sample=True):
    input_load_func = target_load_func = np.load
    transformer = flip
    assert (not remove_empty_sample or not class_balance), 'Remove empty sample might conflict with class balncing'
    
    train_input_samples, train_target_samples = [], []
    for input_root, target_root in zip(input_roots, target_roots):
        train_input_paths = get_files(input_root, keys=train_cases)
        train_target_paths = get_files(target_root, keys=train_cases)

        train_input_samples.extend(train_input_paths)
        train_target_samples.extend(train_target_paths)

    if class_balance and not remove_empty_sample:
        p_input_path, p_target_path = [], []
        n_input_path, n_target_path = [], []
        for input_path, target_path in zip(train_input_samples, train_target_samples):
            if 'positive' in input_path:
                p_input_path.append(input_path)
                p_target_path.append(target_path)
            else:
                n_input_path.append(input_path)
                n_target_path.append(target_path)
            
        np.random.shuffle(p_input_path)
        np.random.shuffle(p_target_path)
        np.random.shuffle(n_input_path)
        np.random.shuffle(n_target_path)
        num_sample = min(len(p_input_path), len(n_input_path))
        train_input_samples = p_input_path[:num_sample] + n_input_path[:num_sample]
        train_target_samples = p_target_path[:num_sample] + n_target_path[:num_sample]

    if remove_empty_sample:
        for input_path, target_path in zip(train_input_samples, train_target_samples):
            if np.sum(np.load(target_path)) <= 0:
                train_input_samples.remove(input_path)
                train_target_samples.remove(target_path)
    
    # print(len(train_target_samples))
    train_dataset = GeneralDataset(
        train_input_samples, train_target_samples, input_load_func, target_load_func, data_transformer=transformer)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)

    def get_samples(roots, cases):
        samples = []
        for root in roots:
            samples.extend(get_files(root, keys=cases))
        return samples

    valid_dataloader = None
    if valid_cases is not None:
        valid_input_samples = get_samples(input_roots, valid_cases)   
        valid_target_samples = get_samples(target_roots, valid_cases)   
        valid_dataset = GeneralDataset(valid_input_samples, valid_target_samples, input_load_func, target_load_func)
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    
    return train_dataloader, valid_dataloader




