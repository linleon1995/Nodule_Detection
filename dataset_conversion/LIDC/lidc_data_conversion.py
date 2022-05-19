from genericpath import exists
import os
from cv2 import split
import numpy as np
import pandas as pd

from data.volume_generator import lidc_nodule_volume_generator
from data.data_utils import get_files
from dataset_conversion.medical_to_img import volumetric_data_preprocess
from dataset_conversion.build_coco import build_lidc_nodule_coco
from dataset_conversion.TMH.tmh_data_conversion import get_cv_split
from utils.train_utils import set_deterministic


def main():
    preprocess_root = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-preprocess'
    raw_vol_path = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data'
    mask_vol_path = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-preprocess\masks_test\3'
    image_path = os.path.join(preprocess_root, 'image')
    coco_path = os.path.join(preprocess_root, 'coco')
    case_pids = None
    num_fold = 5
    case_shuffle = True
    area_threshold = 8
    cat_ids = {'Nodule': 1}
    seed = 1
    height, width= 512, 512

    set_deterministic(seed)

    # Save data in npy

    # # Convert data to image
    # volume_generator_builder = lidc_nodule_volume_generator(
    #     data_path=raw_vol_path, mask_path=mask_vol_path, case_indices=case_pids)
    # volume_generator = volume_generator_builder.build()
    # volumetric_data_preprocess(save_path=image_path, volume_generator=volume_generator)
    
    
    # Save NoduleNet COCO annotation
    NoduleNet_coco_save_path = os.path.join(coco_path, 'NoduleNet')
    os.makedirs(NoduleNet_coco_save_path, exist_ok=True)
    csv_root = rf'C:\Users\test\Desktop\Leon\Projects\NoduleNet\split\cross_val'
    train_csv = get_files(csv_root, 'train.csv')
    valid_csv = get_files(csv_root, 'val.csv')
    total_pid = get_files(os.path.join(image_path, 'Image'), return_fullpath=False, recursive=False, get_dirs=True)

    cv_split_indices = {}
    for i in range(6):
        train_pids = pd.read_csv(train_csv[i], usecols=[0], names=['colA'])
        valid_pids = pd.read_csv(valid_csv[i], usecols=[0], names=['colA'])
        # for pid in valid_pids['colA']:
        #     print(pid)
        train_indices = [total_pid.index(pid) for pid in train_pids['colA']]
        val_indices = [total_pid.index(pid) for pid in valid_pids['colA']]
        cv_split_indices[i] = {'train': [train_indices], 'test': [val_indices]}
        
    for fold in cv_split_indices:
        coco_split_path = os.path.join(NoduleNet_coco_save_path, f'cv-{num_fold}', str(fold))
        if not os.path.isdir(coco_split_path):
            os.makedirs(coco_split_path)

        split_indices = cv_split_indices[fold]
        build_lidc_nodule_coco(
            data_path=image_path, save_path=coco_split_path, split_indices=split_indices, 
            cat_ids=cat_ids, area_threshold=area_threshold, height=height, width=width)

    # # Save data in coco format
    # num_case = len(get_files(os.path.join(image_path, 'Image'), recursive=False, get_dirs=True))
    # cv_split_indices = get_cv_split(num_fold, num_case, shuffle=case_shuffle)

    # for fold in cv_split_indices:
    #     coco_split_path = os.path.join(coco_path, f'cv-{num_fold}', str(fold))
    #     if not os.path.isdir(coco_split_path):
    #         os.makedirs(coco_split_path)

    #     split_indices = cv_split_indices[fold]
    #     build_lidc_nodule_coco(
    #         data_path=image_path, save_path=coco_split_path, split_indices=split_indices, 
    #         cat_ids=cat_ids, area_threshold=area_threshold, height=height, width=width)

if __name__ == '__main__':
    main()