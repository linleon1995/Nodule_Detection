
import os
import numpy as np
from pprint import pprint
import SimpleITK as sitk
import pandas as pd
from data import data_utils
from data.data_utils import load_itk, get_files



def TMH_merging_check(data_path, save_path):
    # benign_root = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule\TMH-Benign\raw'
    # malignant_root = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule\TMH-Malignant\raw'
    # benign_paths = get_files(benign_root, 'mhd')
    # malignant_paths = get_files(malignant_root, 'mhd')
    # total_paths = benign_paths + malignant_paths

    total_paths = get_files(data_path, 'mhd')

    raw_paths, mask_paths = [], []
    for idx, path in enumerate(total_paths):
        if 'raw' in path:
            raw_paths.append(path)
        if 'mask' in path:
            mask_paths.append(path)
    raw_paths.sort()
    mask_paths.sort()
    # raw_paths = raw_paths[:30]
    # mask_paths = mask_paths[:30]
       
    process_list =[]
    df = pd.DataFrame()
    merge_mapping = []
    for idx, (raw_path, mask_path) in enumerate(zip(raw_paths, mask_paths)):
        folder, filename = os.path.split(raw_path)
        _, pid = os.path.split(os.path.split(folder)[0])
        if raw_path in process_list:
            continue
        else:
            process_list.append(raw_path)
        compare_raw_paths = raw_paths.copy()
        compare_raw_paths = list(set(compare_raw_paths)-set(process_list))
        same_list = [pid]
        for compare_path in compare_raw_paths:
            compare_folder, compare_filename = os.path.split(compare_path)
            _, compare_pid = os.path.split(os.path.split(compare_folder)[0])
            same_name, same_value = check_same_volume(raw_path, compare_path)

            if same_name or same_value:
                same_list.append(compare_pid)
                process_list.append(compare_path)
                print(f'pid {pid} compare_pid {compare_pid} name {same_name} value {same_value}')
                
        same_mask_paths = []
        for mask_path in mask_paths:
            for same_pid in same_list:
                if same_pid == os.path.split(os.path.split(os.path.split(mask_path)[0])[0])[1]:
                    if 'm' in same_pid:
                        malignancy = 2
                    elif 'B' in same_pid:
                        malignancy = 1
                    same_mask_paths.append({'path': mask_path, 'malignancy': malignancy})
                    break 
                
        print(same_list)
        merge_mapping.append({'raw': raw_path, 'mask': same_mask_paths})
        row_df = {f'merge_case': f'TMH{idx:03d}'}
        row_df.update({f'case_{same_idx:03d}': pid for same_idx, pid in enumerate(same_list)})
        df = df.append(row_df, ignore_index=True)
        
    df.to_csv(os.path.join(save_path, 'merge_table.csv'), index=False)
    return merge_mapping


def merge_data(merge_mapping, data_path, save_dir, filekey='case'):
    # merge_mapping = TMH_merging_check(data_path, save_dir)

    for vol_idx, repeat_paths in enumerate(merge_mapping, 1):
        raw_path, mask_paths = repeat_paths['raw'], repeat_paths['mask']
        repeat_masks_binary, repeat_masks_semantic, repeat_malignancy = [], [], []
        dirname = f'{filekey}{vol_idx:04d}'
        case_raw_dir = os.path.join(save_dir, dirname, 'raw')
        case_mask_dir = os.path.join(save_dir, dirname, 'mask')
        os.makedirs(case_raw_dir, exist_ok=True)
        os.makedirs(case_mask_dir, exist_ok=True)

        output_image_filename = os.path.join(case_raw_dir, '.'.join(os.path.split(raw_path)[1].split('.')[-2:]))
        # Use the first filename of merging files as the new filename
        output_mask_filename = os.path.join(case_mask_dir, '.'.join(os.path.split(mask_paths[0]['path'])[1].split('.')[-2:]))
        print(f'Merge and Generate new file {dirname}')
        # print(f'Saving merging volume {outputMaskFileName}')

        for mask_path in mask_paths:
            # ct_scan, _, _, _ = data_utils.load_itk(path)
            mask_itkimage = sitk.ReadImage(mask_path['path'])
            mask_vol = sitk.GetArrayFromImage(mask_itkimage)
            repeat_masks_binary.append(mask_vol)
            repeat_malignancy.append(mask_path['malignancy'])
            repeat_masks_semantic.append(mask_vol*mask_path['malignancy'])
        binary_mask_vol = sum(repeat_masks_binary)
        semantic_mask_vol = sum(repeat_masks_semantic)

        if np.sum(binary_mask_vol>1) > 0:
            print('Warning: overlapping among nodules', dirname)
            repeat_area = np.where(binary_mask_vol>1, 1, 0)
            if 2 in repeat_malignancy:
                repeat_area_malignancy = 2
            else:
                repeat_area_malignancy = 1
            semantic_mask_vol[repeat_area==1] = repeat_area_malignancy
        semantic_mask_vol = np.uint8(semantic_mask_vol)

        new_itkimage = data_utils.modify_array_in_itk(mask_itkimage, semantic_mask_vol)
        sitk.WriteImage(new_itkimage, output_mask_filename)

        itkimage = sitk.ReadImage(raw_path)
        sitk.WriteImage(itkimage, output_image_filename)
    print('Merging process complete!\n')


# def merge_data(data_path, save_dir, filekey='case'):
#     total_repeat_paths, repeat_simple = get_merge_paths(data_path)
#     record_merging(total_repeat_paths, save_dir)

#     for vol_idx, repeat_paths in enumerate(total_repeat_paths, 1):
#         raw_path, mask_paths = repeat_paths['raw'], repeat_paths['mask']
#         repeat_ct = []
#         filename = f'1{filekey}{vol_idx:04d}'
#         case_raw_dir = os.path.join(save_dir, filename, 'raw')
#         case_mask_dir = os.path.join(save_dir, filename, 'mask')
#         os.makedirs(case_raw_dir, exist_ok=True)
#         os.makedirs(case_mask_dir, exist_ok=True)

#         outputImageFileName = os.path.join(case_raw_dir, os.path.split(raw_path)[1])
#         # Use the first filename of merging files as the new filename
#         outputMaskFileName = os.path.join(case_mask_dir, os.path.split(mask_paths[0])[1])
#         print(f'Merge and Generate new file {filename}')
#         # print(f'Saving merging volume {outputMaskFileName}')

#         for path in mask_paths:
#             # ct_scan, _, _, _ = data_utils.load_itk(path)
#             itkimage = sitk.ReadImage(path)
#             ct_scan = sitk.GetArrayFromImage(itkimage)
#             repeat_ct.append(ct_scan)
#         merge_ct = sum(repeat_ct)
#         # assert np.sum(merge_ct>1) == 0
#         if np.sum(merge_ct>1) > 0:
#             print('Warning: overlapping among nodules', case_raw_dir)
#         merge_ct = np.where(merge_ct>0, 1, 0)
#         merge_ct = np.uint8(merge_ct)

#         new_itkimage = data_utils.modify_array_in_itk(itkimage, merge_ct)
#         sitk.WriteImage(new_itkimage, outputMaskFileName)

#         image = sitk.ReadImage(raw_path)
#         ct_scan = sitk.GetArrayFromImage(itkimage)
#         sitk.WriteImage(image, outputImageFileName)
#     print('Merging process complete!\n')


# def record_merging(total_repeat_paths, save_dir):
#     merge_mapping = {}
#     for paths in total_repeat_paths:
#         raw_filename = os.path.split(paths['raw'])[1][:-4]
#         mask_filenames = [os.path.split(path)[1][:-4] for path in paths['mask']]
#         merge_mapping[raw_filename] = mask_filenames
#     merge_table = pd.DataFrame.from_dict(merge_mapping, orient='index')
#     merge_table.to_csv(os.path.join(save_dir, 'merge_table.csv'))


def check_same_volume(vol1_path, vol2_path):
    if vol1_path == vol2_path:
        same_name = True
    else:
        same_name = False
        
    vol1, _, _, _ = load_itk(vol1_path)
    vol2, _, _, _ = load_itk(vol2_path)

    same_value = False
    if vol1.shape == vol2.shape:
        if (vol1 == vol2).all():
            same_value = True

    return same_name, same_value
    

# def get_merge_paths(data_path):
#     # Merger the same case of ASUS-Nodule in a single mhd file
#     fullpath_list = data_utils.get_files(data_path, 'mhd')
#     raw_fullpaths, mask_fullpaths = [], []
#     raw_filenames, mask_filenames = [], []
#     for path in fullpath_list:
#         root_path, filename = os.path.split(path)
#         dir_name = os.path.split(root_path)[1]
#         if 'raw' in dir_name:
#             raw_fullpaths.append(path)
#             raw_filenames.append(os.path.split(path)[1])
#         if 'mask' in dir_name:
#             mask_fullpaths.append(path)
#             mask_filenames.append(os.path.split(path)[1])
   
#     unique_raw_filenames = [raw_filenames[0]]
#     for idx in range(1, len(raw_filenames)):
#         if raw_filenames[idx] != raw_filenames[idx-1]:
#             unique_raw_filenames.append(raw_filenames[idx])

#     total_repeat_paths, total_repeat_paths_simple = [], []
#     for filename in unique_raw_filenames:
#         repeat_paths, repeat_paths_simple = [], []
#         repeat_path_pair, repeat_path_pair_simple = {}, {}
#         for raw_fullpath, mask_fullpath, compare_filename in zip(raw_fullpaths, mask_fullpaths, raw_filenames):
#             if filename == compare_filename:
#                 repeat_paths.append(mask_fullpath)
#                 repeat_paths_simple.append(os.path.split(os.path.split(mask_fullpath)[0])[1])
#                 if 'raw' not in repeat_path_pair:
#                     repeat_path_pair['raw'] = raw_fullpath
#                     repeat_path_pair_simple['raw'] = os.path.split(os.path.split(raw_fullpath)[0])[1]

#         repeat_path_pair['mask'] = repeat_paths
#         repeat_path_pair_simple['mask'] = repeat_paths_simple
#         total_repeat_paths.append(repeat_path_pair)
#         total_repeat_paths_simple.append(repeat_path_pair_simple)
#     # print(total_repeat_paths)
#     return total_repeat_paths, total_repeat_paths_simple


# if __name__ == '__main__':
#     DATA_PATH_B = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\benign'
#     SAVE_PATH_B = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\benign_merge'
#     DATA_PATH_M = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant'
#     SAVE_PATH_M = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant_merge'

#     for data_path, save_dir in zip((DATA_PATH_B, DATA_PATH_M), (SAVE_PATH_B, SAVE_PATH_M)):
#         if os.path.split(data_path)[1] == 'benign':
#             filekey = 'B'
#         elif os.path.split(data_path)[1] == 'malignant':
#             filekey = 'm'

#         merge_asus_data(data_path=data_path, save_dir=save_dir, filekey=filekey)