import os

from data import medical_to_img
from data import build_coco
from data import asus_data_merge

import site_path
from modules.utils import configuration
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
from utils.volume_generator import luna16_volume_generator, asus_nodule_volume_generator

DATASET_NAME = ['Benign', 'Malignant'] # Benign, Malignant, LUNA16, LUNA16-Round
BENIGN_TRAIN_SPLIT = list(range(17))      
BENIGN_VALID_SPLIT = list(range(17, 19))   
BENIGN_TEST_SPLIT = list(range(19, 25))      

MALIGNANT_TRAIN_SPLIT = list(range(34))   
MALIGNANT_VALID_SPLIT = list(range(34, 36)) 
MALIGNANT_TEST_SPLIT = list(range(36, 44))  

# TODO: Add some logging file or excel for recording   
# TODO: coco


def get_asus_data_split(dataset_name):
    if dataset_name == 'Benign':
        train_split = BENIGN_TRAIN_SPLIT
        valid_split = BENIGN_VALID_SPLIT
        test_split = BENIGN_TEST_SPLIT
    elif dataset_name == 'Malignant':
        train_split = MALIGNANT_TRAIN_SPLIT
        valid_split = MALIGNANT_VALID_SPLIT
        test_split = MALIGNANT_TEST_SPLIT
    else:
        raise ValueError('Unknown dataset name.')
    
    output_data_split = {}
    data_split = {'train': train_split, 'valid': valid_split, 'test': test_split}
    for split in data_split:
        for idx in data_split[split]:
            output_data_split[idx] = split

    return output_data_split
        

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


def build_parameters(dataset_name):
    if dataset_name not in ['Benign', 'Malignant']:
        return None

    config_name = f'ASUS-{dataset_name}'
    cfg = configuration.load_config(f'data/config/{config_name}.yml', dict_as_member=True)
    data_root = cfg.PATH.DATA_ROOT
    task_name = cfg.TASK_NAME
    cat_ids = cfg.CATEGORY_ID[task_name]
    area_threshold = cfg.AREA_THRESHOLD

    merge_path = os.path.join(os.path.split(data_root)[0], 'merge')
    image_path = os.path.join(os.path.split(data_root)[0], 'image')
    coco_path = os.path.join(os.path.split(data_root)[0], 'coco', task_name)
    data_split = get_asus_data_split(dataset_name)
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
        'data_root': data_root,
        'cat_ids': cat_ids,
        'area_threshold': area_threshold,
        'merge_path': merge_path,
        'image_path': image_path,
        'coco_path': coco_path,
        'data_split': data_split,
        'category': category,
        'split_indices': split_indices
    }

    # coco_parameters = {
    #     'data_name': dataset_name,
    #     'area_threshold': area_threshold, 
    #     'cat_ids': cat_ids, 
    #     'data_path': image_path, 
    #     'save_path': coco_path,
    #     'split_indices': split_indices,
    #     'category': category
    # }
    return data_parameters
    

def data_preprocess(dataset_names):
    for dataset_name in dataset_names:
        dataset_parameter = build_parameters(dataset_name)

        if dataset_parameter is not None:
            data_root = dataset_parameter['data_root']
            merge_path = dataset_parameter['merge_path']
            image_path = dataset_parameter['image_path']
            coco_path = dataset_parameter['coco_path']
            # volume_generator = dataset_parameter['volume_generator']
            data_split = dataset_parameter['data_split']
            split_indices = dataset_parameter['split_indices']
            cat_ids = dataset_parameter['cat_ids']
            area_threshold = dataset_parameter['area_threshold']
            category = dataset_parameter['category']
            kc_image_path = image_path.replace('image', 'kc_image')

            for path in [merge_path, image_path, coco_path, kc_image_path]:
                if not os.path.isdir(path):
                    os.makedirs(path)

            # Merge mhd data
            if dataset_name == 'Benign':
                filekey = 'B'
            elif dataset_name == 'Malignant':
                filekey = 'm'
            # asus_data_merge.merge_asus_data(data_root, merge_path, filekey)

            # # Convert medical 3d volume data to image format
            # volume_generator = asus_nodule_volume_generator(nodule_type=dataset_name, data_path=merge_path)
            # medical_to_img.volumetric_data_preprocess(save_path=image_path, volume_generator=volume_generator)
            # volume_generator = asus_nodule_volume_generator(nodule_type=dataset_name, data_path=merge_path)
            # medical_to_img.volumetric_data_preprocess_KC(data_split, save_path=kc_image_path, volume_generator=volume_generator)
            
            # Build up coco-structure
            build_coco.build_asus_nodule_coco(
                dataset_name, data_path=image_path, save_path=coco_path, split_indices=split_indices, 
                cat_ids=cat_ids, area_threshold=area_threshold, category=category)


def main():
    data_preprocess(DATASET_NAME)


if __name__ == '__main__':
    main()
    pass