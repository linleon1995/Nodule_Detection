
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from collections import OrderedDict
from data.data_utils import load_itk, get_files, modify_array_in_itk


def merge_data(merge_mapping, data_path, save_dir, filekey='case'):
    # merge_mapping = TMH_merging_check(data_path, save_dir)
    total_nodule = 0
    for vol_idx, repeat_paths in enumerate(merge_mapping, 1):
        if vol_idx not in [2, 42, 78]:
            continue
        raw_path, mask_paths, filename = repeat_paths['raw'], repeat_paths['mask'], repeat_paths['filename']
        repeat_masks_binary, repeat_masks_semantic, repeat_malignancy = [], [], []
        case_raw_dir = os.path.join(save_dir, filename, 'raw')
        case_mask_dir = os.path.join(save_dir, filename, 'mask')
        os.makedirs(case_raw_dir, exist_ok=True)
        os.makedirs(case_mask_dir, exist_ok=True)

        output_image_filename = os.path.join(case_raw_dir, '.'.join(os.path.split(raw_path)[1].split('.')[-2:]))
        # Use the first filename of merging files as the new filename
        output_mask_filename = os.path.join(case_mask_dir, '.'.join(os.path.split(mask_paths[0]['path'])[1].split('.')[-2:]))
        print(f'Merge and Generate new file {filename}')
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

        overlapping = np.sum(binary_mask_vol>1)
        if overlapping > 0:
            print('Warning: overlapping among nodules', filename)
            repeat_area = np.where(binary_mask_vol>1, 1, 0)
            if 2 in repeat_malignancy:
                repeat_area_malignancy = 2
            else:
                repeat_area_malignancy = 1
            semantic_mask_vol[repeat_area==1] = repeat_area_malignancy
        semantic_mask_vol = np.uint8(semantic_mask_vol)

        import cc3d
        nodule_num = np.unique(cc3d.connected_components(semantic_mask_vol, connectivity=26)).size-1
        total_nodule += nodule_num

        # new_itkimage = modify_array_in_itk(mask_itkimage, semantic_mask_vol)
        # sitk.WriteImage(new_itkimage, output_mask_filename)

        # itkimage = sitk.ReadImage(raw_path)
        # sitk.WriteImage(itkimage, output_image_filename)
    print('Merging process complete!\n')


def TMH_merging_check(data_path, save_path):
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
    row_df_list = []
    merge_idx = 1
    
    print('Checking TMH data merging cases')
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
        
        filename = f'TMH{merge_idx:04d}'
        print(f'{merge_idx} {same_list}')
        merge_mapping.append({'filename': filename, 'raw': raw_path, 'mask': same_mask_paths})
        # row_df = {f'output': f'TMH{merge_idx:03d}'}
        # row_df.update({f'merge_case_{same_idx:03d}': pid for same_idx, pid in enumerate(same_list)})
        row_df = OrderedDict()
        row_df['output'] = filename
        for same_idx, pid in enumerate(same_list):
            row_df[f'merge_case_{same_idx:03d}'] = pid
        # df = df.append(row_df, ignore_index=True)
        row_df_list.append(pd.DataFrame(row_df, index=[0]))
        merge_idx += 1
    df = pd.concat(row_df_list)    
    df.to_csv(os.path.join(save_path, 'merge_table.csv'), index=False)
    return merge_mapping


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
