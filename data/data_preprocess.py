import os
import numpy as np

from data import medical_to_img
from data import build_coco
from data import asus_data_merge
from data import data_utils

import site_path
from modules.utils import configuration
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
from data.volume_generator import luna16_volume_generator, asus_nodule_volume_generator

DATASET_NAME = ['Benign', 'Malignant'] # Benign, Malignant, LUNA16, LUNA16-Round
# BENIGN_TRAIN_SPLIT = list(range(17))      
# BENIGN_VALID_SPLIT = list(range(17, 19))   
# BENIGN_TEST_SPLIT = list(range(19, 25))      

# MALIGNANT_TRAIN_SPLIT = list(range(34))   
# MALIGNANT_VALID_SPLIT = list(range(34, 36)) 
# MALIGNANT_TEST_SPLIT = list(range(36, 44))  

# TODO: Add some logging file or excel for recording   
# TODO: coco


# def get_asus_data_split(dataset_name):
#     if dataset_name == 'Benign':
#         train_split = BENIGN_TRAIN_SPLIT
#         valid_split = BENIGN_VALID_SPLIT
#         test_split = BENIGN_TEST_SPLIT
#     elif dataset_name == 'Malignant':
#         train_split = MALIGNANT_TRAIN_SPLIT
#         valid_split = MALIGNANT_VALID_SPLIT
#         test_split = MALIGNANT_TEST_SPLIT
#     else:
#         raise ValueError('Unknown dataset name.')
    
#     output_data_split = {}
#     data_split = {'train': train_split, 'valid': valid_split, 'test': test_split}
#     for split in data_split:
#         for idx in data_split[split]:
#             output_data_split[idx] = split

#     return output_data_split
        

# def convert_parameter(dataset_name, area_threshold, cat_ids, data_path, save_path, split_indices):
#     if dataset_name == 'Benign':
#         split_indices = {
#             'train': BENIGN_TRAIN_INDICES,
#             'valid': BENIGN_VALID_INDICES,
#             'test': BENIGN_TEST_INDICES,
#         }
#     elif dataset_name == 'Malignant':
#         split_indices = {
#             'train': MALIGNANT_TRAIN_INDICES,
#             'valid': MALIGNANT_VALID_INDICES,
#             'test': MALIGNANT_TEST_INDICES,
#         }

#     return {
#         'dataset_name': dataset_name,
#         'area_threshold': area_threshold, 
#         'cat_ids': cat_ids, 
#         'data_path': data_path, 
#         'save_path': save_path,
#         'split_indices': split_indices
#     }

# def build_coco_parameters():
#     # Benign
#     name = 'benign'
#     benign_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\benign\raw_merge'
#     train_indices = list(range(0, 17))
#     valid_indices = list(range(17, 19))
#     test_indices = list(range(19, 25))
#     split_indices = {'train': train_indices,
#                      'valid': valid_indices,
#                      'test': test_indices}
#     benign
#     benign_convert_info = DatasetConvertInfo(
#         name, benign_path, split_indices, area_threshold, cat_ids, save_path)
    
#     # Malignant
#     name = 'malignant'
#     malignant_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\malignant\raw_merge'
#     train_indices = list(range(0, 34))
#     valid_indices = list(range(34, 36))
#     test_indices = list(range(36, 44))
#     split_indices = {'train': train_indices,
#                      'valid': valid_indices,
#                      'test': test_indices}
#     malignant_convert_info = DatasetConvertInfo(
#         name, malignant_path, split_indices, area_threshold, cat_ids, save_path)

#     coco_parameters = {}
#     return coco_parameters


def get_cv_split(num_fold, num_sample, shuffle=False):
    assert num_fold > 0 and num_sample > 0, 'The fold number and sample number should both bigger than 0'
    assert num_sample > num_fold, 'The fold number should not bigger than sample number'
    num_sample_in_fold = num_sample // num_fold
    remain = num_sample - num_fold * num_sample_in_fold
    base_num = [num_sample_in_fold+1  if i <= remain-1 else num_sample_in_fold for i in range(num_fold)]
    sample_indices = list(range(num_sample))
    if shuffle:
        np.random.shuffle(sample_indices)

    indices = []
    acc_num = 0
    for num in base_num:
        indices.append(list(sample_indices[acc_num:acc_num+num]))
        acc_num += num

    cv_split = {}
    for fold in range(num_fold):
        test_slice = slice(fold, fold+1)
        train_slices = [slice(0, fold), slice(fold+1, num_fold)]
        train_indices = []
        for train_slice in train_slices:
            train_indices.extend(indices[train_slice])
        cv_split[fold] = {'train': train_indices, 'test': indices[test_slice]}
    return cv_split


# def get_data_cv_split():
#     cv_split = {}
#     for fold in range(1, num_fold+1):
#         cv_split[fold] = {'train': (),}


def build_parameters(dataset_name):
    if dataset_name not in ['Benign', 'Malignant']:
        return None

    config_name = f'ASUS-{dataset_name}'
    cfg = configuration.load_config(f'data/config/{config_name}.yml', dict_as_member=True)
    data_root = cfg.PATH.DATA_ROOT
    task_name = cfg.TASK_NAME
    cat_ids = cfg.CATEGORY_ID[task_name]
    area_threshold = cfg.AREA_THRESHOLD
    data_split = cfg.SPLIT
    num_fold = cfg.CROSS_VALID_FOLD

    raw_path = os.path.join(data_root, 'raw')
    merge_path = os.path.join(data_root, 'merge')
    image_path = os.path.join(data_root, 'image')
    coco_path = os.path.join(data_root, 'coco', task_name)
    # data_split = get_asus_data_split(dataset_name)
    if task_name == 'Nodule_Detection':
        category = 'Nodule'
    elif task_name == 'Malignancy':
        category = dataset_name

    split_indices = {}
    for split_idx, split_name in data_split.items():
        if split_name in split_indices:
            split_indices[split_name].append(split_idx)
        else:
            split_indices[split_name] = [split_idx]

    data_parameters = {
        'raw_path': raw_path,
        'cat_ids': cat_ids,
        'area_threshold': area_threshold,
        'merge_path': merge_path,
        'image_path': image_path,
        'coco_path': coco_path,
        # 'cv_split_indices': cv_split_indices,
        # 'data_split': data_split,
        'category': category,
        # 'split_indices': split_indices,
        'num_fold': num_fold,
    }

    return data_parameters
    

def data_preprocess(dataset_names):
    for dataset_name in dataset_names:
        dataset_parameter = build_parameters(dataset_name)

        if dataset_parameter is not None:
            raw_path = dataset_parameter['raw_path']
            merge_path = dataset_parameter['merge_path']
            image_path = dataset_parameter['image_path']
            coco_path = dataset_parameter['coco_path']
            # volume_generator = dataset_parameter['volume_generator']
            # data_split = dataset_parameter['data_split']
            # split_indices = dataset_parameter['split_indices']
            # cv_split_indices = dataset_parameter['cv_split_indices']
            cat_ids = dataset_parameter['cat_ids']
            area_threshold = dataset_parameter['area_threshold']
            category = dataset_parameter['category']
            num_fold = dataset_parameter['num_fold']
            shuffle = False
            kc_image_path = image_path.replace('image', 'kc_image')

            for path in [merge_path, image_path, kc_image_path]:
                if not os.path.isdir(path):
                    os.makedirs(path)

            # Merge mhd data
            if dataset_name == 'Benign':
                filekey = 'B'
            elif dataset_name == 'Malignant':
                filekey = 'm'
            # asus_data_merge.merge_asus_data(raw_path, merge_path, filekey)

            # # Convert medical 3d volume data to image format
            # volume_generator = asus_nodule_volume_generator(nodule_type=f'ASUS-{dataset_name}', data_path=merge_path)
            # medical_to_img.volumetric_data_preprocess(save_path=image_path, volume_generator=volume_generator)
            # volume_generator = asus_nodule_volume_generator(nodule_type=f'ASUS-{dataset_name}', data_path=merge_path)
            # medical_to_img.volumetric_data_preprocess_KC(data_split, save_path=kc_image_path, volume_generator=volume_generator)
            
            # Build up coco-structure
            num_case = len(data_utils.get_files(merge_path, recursive=False, get_dirs=True))
            cv_split_indices = get_cv_split(num_fold, num_case, shuffle)
                
            for fold in cv_split_indices:
                coco_split_path = os.path.join(coco_path, f'cv-{num_fold}', str(fold))
                if not os.path.isdir(coco_split_path):
                    os.makedirs(coco_split_path)

                split_indices = cv_split_indices[fold]
                build_coco.build_asus_nodule_coco(
                    data_path=image_path, save_path=coco_split_path, split_indices=split_indices, 
                    cat_ids=cat_ids, area_threshold=area_threshold, category=category)


def main():
    data_preprocess(DATASET_NAME)


if __name__ == '__main__':
    main()
    pass