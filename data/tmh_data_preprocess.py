import os
import numpy as np

from data import medical_to_img
from data import build_coco
from data import asus_data_merge
from data import data_utils
from data.data_analysis import TMH_nodule_base_check, TMH_merging_check

import site_path
from modules.utils import configuration
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
from data.volume_generator import luna16_volume_generator, asus_nodule_volume_generator

DATASET_NAME = ['Malignant', 'Benign'] # Benign, Malignant, LUNA16, LUNA16-Round


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


def build_parameters(dataset_name):
    if dataset_name not in ['Benign', 'Malignant']:
        return None

    config_name = f'TMH-{dataset_name}'
    cfg = configuration.load_config(f'data/config/{config_name}.yml', dict_as_member=True)
    data_root = cfg.PATH.DATA_ROOT
    task_name = cfg.TASK_NAME
    cat_ids = cfg.CATEGORY_ID[task_name]
    area_threshold = cfg.AREA_THRESHOLD
    data_split = cfg.SPLIT
    num_fold = cfg.CROSS_VALID_FOLD

    raw_path = os.path.join(data_root, 'raw')
    stats_path = os.path.join(data_root, 'stats_path')
    merge_path = os.path.join(data_root, 'merge')
    image_path = os.path.join(data_root, 'image')
    coco_path = os.path.join(data_root, 'coco', task_name)
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
        'stats_path': stats_path,
        'merge_path': merge_path,
        'image_path': image_path,
        'coco_path': coco_path,
        'category': category,
        'num_fold': num_fold,
    }

    return data_parameters
    

def data_preprocess(dataset_names):
    for dataset_name in dataset_names:
        dataset_parameter = build_parameters(dataset_name)

        if dataset_parameter is not None:
            raw_path = dataset_parameter['raw_path']
            stats_path = dataset_parameter['stats_path']
            merge_path = dataset_parameter['merge_path']
            image_path = dataset_parameter['image_path']
            coco_path = dataset_parameter['coco_path']
            cat_ids = dataset_parameter['cat_ids']
            area_threshold = dataset_parameter['area_threshold']
            category = dataset_parameter['category']
            num_fold = dataset_parameter['num_fold']
            shuffle = False
            kc_image_path = image_path.replace('image', 'kc_image')

            for path in [merge_path, image_path, kc_image_path, stats_path]:
                os.makedirs(path, exist_ok=True)

            if dataset_name == 'Benign':
                filekey = 'B'
                case_pids = [f'1B00{i}' for i in range(38, 39)]
                case_pids += [f'1B00{i}' for i in range(53, 54)]
                case_pids += [f'1B00{i}' for i in range(56, 61)]
                case_pids = None
            elif dataset_name == 'Malignant':
                filekey = 'm'
                case_pids = [f'1m00{i}' for i in range(58, 60)]
                case_pids = None

            # # Merge mhd data
            # asus_data_merge.merge_asus_data(raw_path, merge_path, filekey)

            # Convert medical 3d volume data to image format
            # volume_generator = asus_nodule_volume_generator(data_path=raw_path, case_pids=case_pids)
            # medical_to_img.volumetric_data_preprocess(save_path=image_path, volume_generator=volume_generator)
            # volume_generator = asus_nodule_volume_generator(data_path=merge_path)
            # medical_to_img.volumetric_data_preprocess_KC(data_split, save_path=kc_image_path, volume_generator=volume_generator)

            # TMH base check
            volume_generator = asus_nodule_volume_generator(data_path=raw_path, case_pids=case_pids)
            TMH_nodule_base_check(volume_generator, save_path=stats_path)

            # # Build up coco-structure
            # num_case = len(data_utils.get_files(merge_path, recursive=False, get_dirs=True))
            # cv_split_indices = get_cv_split(num_fold, num_case, shuffle)
                
            # for fold in cv_split_indices:
            #     coco_split_path = os.path.join(coco_path, f'cv-{num_fold}', str(fold))
            #     if not os.path.isdir(coco_split_path):
            #         os.makedirs(coco_split_path)

            #     split_indices = cv_split_indices[fold]
            #     build_coco.build_asus_nodule_coco(
            #         data_path=image_path, save_path=coco_split_path, split_indices=split_indices, 
            #         cat_ids=cat_ids, area_threshold=area_threshold, category=category)


def main():
    # data_preprocess(DATASET_NAME)
    TMH_merging_check()


if __name__ == '__main__':
    main()
    # pass

    from data.data_utils import load_itk
    import matplotlib.pyplot as plt
    x, _, _, _ = load_itk(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\history\0418_problem_data\1m0043\1m0043raw mhd\1.2.826.0.1.3680043.2.1125.1.66267488139869463859646041266078917.mhd')
    y, _, _, _ = load_itk(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\history\0418_problem_data\1m0043\1m0043mask mhd\1.2.826.0.1.3680043.2.1125.1.20492007384673651600845318549231386.mhd')
    # print(x.shape)
    # print(np.sum(y))

    # fig, ax = plt.subplots(1,1)
    # for i in range(x.shape[0]):
    #     if np.sum(y[i])>0:
    #         print(i)
    #         ax.imshow(x[i], 'gray')
    #         ax.imshow(y[i], alpha=0.2)
    #         fig.savefig(f'img{i}.png')
    #         plt.cla()
        