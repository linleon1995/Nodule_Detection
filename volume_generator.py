
import os
import numpy as np
import matplotlib.pyplot as plt
from statistics import median_high
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import cv2_imshow, calculate_malignancy, segment_lung, mask_preproccess, raw_preprocess
from convert_to_coco_structure import lidc_to_datacatlog_valid
import logging
import pandas as pd
from LUNA16_test import dataset_seg
logging.basicConfig(level=logging.INFO)

from modules.data import dataset_utils


def lidc_volume_generator(data_path, case_indices, only_nodule_slices=False):
    case_list = dataset_utils.get_files(data_path, recursive=False, get_dirs=True)
    case_list = np.array(case_list)[case_indices]
    for case_dir in case_list:
        pid = os.path.split(case_dir)[1]
        scan_list = dataset_utils.get_files(os.path.join(case_dir, rf'Image\lung\vol\npy'), 'npy')
        for scan_idx, scan_path in enumerate(scan_list):
            vol = np.load(scan_path)
            mask_vol = np.load(os.path.join(case_dir, rf'Mask\vol\npy', os.path.split(scan_path)[1]))
            mask_vol = np.where(mask_vol>=1, 1, 0)
            if only_nodule_slices:
                nodule_slice_indices = np.where(np.sum(mask_vol, axis=(0,1)))[0]
                vol = vol[...,nodule_slice_indices]
                mask_vol = mask_vol[...,nodule_slice_indices]
            infos = {'pid': pid, 'scan_idx': scan_idx}
            yield vol, mask_vol, infos


def asus_nodule_volume_generator(data_path, case_indices, only_nodule_slices=False):
    case_list = dataset_utils.get_files(data_path, recursive=False, get_dirs=True)
    case_list = np.array(case_list)[case_indices]
    for case_dir in case_list:
        raw_and_mask = dataset_utils.get_files(case_dir, recursive=False, get_dirs=True)
        assert len(raw_and_mask) == 2
        for _dir in raw_and_mask:
            if 'raw' in _dir:
                vol_path = dataset_utils.get_files(_dir, 'mhd', recursive=False)[0]
                vol, _, _ = dataset_utils.load_itk(vol_path)
                vol = np.swapaxes(np.swapaxes(vol, 0, 1), 1, 2)
                for img_idx in range(vol.shape[2]):
                    img = vol[...,img_idx]
                    img = raw_preprocess(img, change_channel=False)
                    # img = segment_lung(img)
                    # if np.max(img)==np.min(img):
                    #     img = np.zeros_like(img)
                    # else:
                    #     img = np.uint8(255*((img-np.min(img))/(np.max(img)-np.min(img))))
                    vol[...,img_idx] = img
            if 'mask' in _dir:
                vol_mask_path = dataset_utils.get_files(_dir, 'mhd', recursive=False)[0]
                mask_vol, _, _ = dataset_utils.load_itk(vol_mask_path)
                mask_vol = np.swapaxes(np.swapaxes(mask_vol, 0, 1), 1, 2)
        pid = os.path.split(case_dir)[1]
        infos = {'pid': pid, 'scan_idx': 0}
        mask_vol = np.where(mask_vol>=1, 1, 0)
        yield vol, mask_vol, infos


def luna16_to_lidc(path, key):
    df = pd.read_csv(path)
    index = df.index
    aa = df['Study UID']
    condition = df['Study UID'] == key
    indices = index[condition]
    return indices['Subject ID']



def luna16_volume_generator(data_path, case_indices=None, only_nodule_slices=False):
    subset_list = dataset_utils.get_files(data_path, 'subset', recursive=False, get_dirs=True)
    subset_list = subset_list[:1]
    
    for subset_dir in subset_list:
        case_list = dataset_utils.get_files(subset_dir, 'mhd', recursive=False)
        if case_indices:
            case_list = case_list[case_indices]
        for case_dir in case_list:
            # TODO: below same with asus-nodules
            series_uid = os.path.split(case_dir)[1][:-4]
            ct = dataset_seg.getCt(series_uid)
            vol = ct.hu_a
            mask_vol = ct.positive_mask
            
            # preprocess
            # TODO: Finish preprocessing part (same with Liwei's code)
            # TODO: check the shape and dims
            infos = {'pid': series_uid, 'scan_idx': 0}
            yield vol, mask_vol, infos
            
            
# def luna16_volume_generator(data_path, case_indices, only_nodule_slices=False):
#     subset_list = dataset_utils.get_files(data_path, 'subset', recursive=False, get_dirs=True)
#     subset_list = subset_list[:1]
    
#     for subset_dir in subset_list:
#         case_list = dataset_utils.get_files(subset_dir, 'mhd', recursive=False)
#         case_list = case_list[case_indices]
#         for case_dir in case_list:
#             # TODO: below same with asus-nodules
#             luna16_name = os.path.split(case_dir)[1][:-4]
#             lidc_name = luna16_to_lidc(os.path.join(data_path, rf'LIDC-IDRI_MetaData.csv'), key=luna16_name)
#             raw_and_mask = dataset_utils.get_files(case_dir, recursive=False, get_dirs=True)
#             assert len(raw_and_mask) == 2
#             for _dir in raw_and_mask:
#                 if 'raw' in _dir:
#                     vol_path = dataset_utils.get_files(_dir, 'mhd', recursive=False)[0]
#                     vol, _, _ = dataset_utils.load_itk(vol_path)
#                     vol = np.swapaxes(np.swapaxes(vol, 0, 1), 1, 2)
#                     for img_idx in range(vol.shape[2]):
#                         img = vol[...,img_idx]
#                         img = raw_preprocess(img, change_channel=False)
#                         vol[...,img_idx] = img
#                 if 'mask' in _dir:
#                     vol_mask_path = dataset_utils.get_files(_dir, 'mhd', recursive=False)[0]
#                     mask_vol, _, _ = dataset_utils.load_itk(vol_mask_path)
#                     mask_vol = np.swapaxes(np.swapaxes(mask_vol, 0, 1), 1, 2)
#             pid = os.path.split(case_dir)[1]
#             infos = {'pid': pid, 'scan_idx': 0}
#             mask_vol = np.where(mask_vol>=1, 1, 0)
#             yield vol, mask_vol, infos
