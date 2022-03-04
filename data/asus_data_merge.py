
import os
import numpy as np
from pprint import pprint
import SimpleITK as sitk

import site_path
from modules.data import dataset_utils


def get_merge_paths(data_path):
    # Merger the same case of ASUS-Nodule in a single mhd file
    fullpath_list = dataset_utils.get_files(data_path, 'mhd')
    raw_fullpaths, mask_fullpaths = [], []
    raw_filenames, mask_filenames = [], []
    for path in fullpath_list:
        root_path, filename = os.path.split(path)
        dir_name = os.path.split(root_path)[1]
        if 'raw' in dir_name:
            raw_fullpaths.append(path)
            raw_filenames.append(os.path.split(path)[1])
        if 'mask' in dir_name:
            mask_fullpaths.append(path)
            mask_filenames.append(os.path.split(path)[1])
   
    unique_raw_filenames = [raw_filenames[0]]
    for idx in range(1, len(raw_filenames)):
        if raw_filenames[idx] != raw_filenames[idx-1]:
            unique_raw_filenames.append(raw_filenames[idx])

    total_repeat_paths = []

    for filename in unique_raw_filenames:
        repeat_paths = []
        repeat_path_pair = {}
        for raw_fullpath, mask_fullpath, compare_filename in zip(raw_fullpaths, mask_fullpaths, raw_filenames):
            if filename == compare_filename:
                repeat_paths.append(mask_fullpath)
                if 'raw' not in repeat_path_pair:
                    repeat_path_pair['raw'] = raw_fullpath

        repeat_path_pair['mask'] = repeat_paths
        total_repeat_paths.append(repeat_path_pair)
    # print(total_repeat_paths)
    return total_repeat_paths


def merge_asus_data(data_path, save_dir, filekey='case'):
    total_repeat_paths = get_merge_paths(data_path)
    
    for vol_idx, repeat_paths in enumerate(total_repeat_paths, 1):
        raw_path, mask_paths = repeat_paths['raw'], repeat_paths['mask']
        repeat_ct = []
        filename = f'1{filekey}{vol_idx:04d}'
        case_raw_dir = os.path.join(save_dir, filename, 'raw')
        case_mask_dir = os.path.join(save_dir, filename, 'mask')
        if not os.path.isdir(case_raw_dir):
            os.makedirs(case_raw_dir)
        if not os.path.isdir(case_mask_dir):
            os.makedirs(case_mask_dir)

        outputImageFileName = os.path.join(case_raw_dir, os.path.split(raw_path)[1])
        outputMaskFileName = os.path.join(case_mask_dir, os.path.split(mask_paths[0])[1])
        print(filename)
        # print(f'Saving merging volume {outputMaskFileName}')

        for path in mask_paths:
            # ct_scan, _, _, _ = dataset_utils.load_itk(path)
            itkimage = sitk.ReadImage(path)
            ct_scan = sitk.GetArrayFromImage(itkimage)
            repeat_ct.append(ct_scan)
        merge_ct = sum(repeat_ct)
        # assert np.sum(merge_ct>1) == 0
        if np.sum(merge_ct>1) > 0:
            print('error', case_raw_dir)
        merge_ct = np.where(merge_ct>0, 1, 0)
        merge_ct = np.uint8(merge_ct)

        new_itkimage = dataset_utils.modify_array_in_itk(itkimage, merge_ct)
        sitk.WriteImage(new_itkimage, outputMaskFileName)

        image = sitk.ReadImage(raw_path)
        ct_scan = sitk.GetArrayFromImage(itkimage)
        sitk.WriteImage(image, outputImageFileName)
        

if __name__ == '__main__':
    DATA_PATH_B = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\benign'
    SAVE_PATH_B = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\benign_merge'
    DATA_PATH_M = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant'
    SAVE_PATH_M = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant_merge'

    for data_path, save_dir in zip((DATA_PATH_B, DATA_PATH_M), (SAVE_PATH_B, SAVE_PATH_M)):
        if os.path.split(data_path)[1] == 'benign':
            filekey = 'B'
        elif os.path.split(data_path)[1] == 'malignant':
            filekey = 'm'

        merge_asus_data(data_path=data_path, save_dir=save_dir, filekey=filekey)