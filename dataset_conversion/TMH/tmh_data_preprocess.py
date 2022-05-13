import os
import numpy as np
import cc3d
import SimpleITK as sitk
from dataset_conversion import medical_to_img
from dataset_conversion.build_coco import build_tmh_nodule_coco
from dataset_conversion.TMH import tmh_data_merge
from dataset_conversion.data_analysis import TMH_nodule_base_check
from data.volume_generator import asus_nodule_volume_generator
from data.data_utils import modify_array_in_itk, get_files

from utils.configuration import load_config
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# DATASET_NAME = 'TMH-Nodule'
CONFIG_PATH = 'dataset_conversion/config/TMH-Nodule.yml'
    

def data_preprocess(dataset_name):
    dataset_parameter = build_parameters(dataset_name)

    data_root = dataset_parameter['data_root']
    raw_path = dataset_parameter['raw_path']
    stats_path = dataset_parameter['stats_path']
    merge_path = dataset_parameter['merge_path']
    image_path = dataset_parameter['image_path']
    coco_path = dataset_parameter['coco_path']
    cat_ids = dataset_parameter['cat_ids']
    area_threshold = dataset_parameter['area_threshold']
    category = dataset_parameter['category']
    num_fold = dataset_parameter['num_fold']
    shuffle = dataset_parameter['shuffle']
    case_pids = dataset_parameter['case_pids']
    kc_image_path = image_path.replace('image', 'kc_image')
    for path in [merge_path, image_path, kc_image_path, stats_path]:
        os.makedirs(path, exist_ok=True)
    
    # # TMH base check
    # raw_paths = get_files(data_root, recursive=False, get_dirs=True)
    # volume_generator = asus_nodule_volume_generator(data_path=raw_paths, case_pids=case_pids)
    # TMH_nodule_base_check(volume_generator, save_path=stats_path)

    # # Merge mhd data
    # merge_mapping = tmh_data_merge.TMH_merging_check(data_root, merge_path)
    # tmh_data_merge.merge_data(merge_mapping, data_root, merge_path, filekey='TMH')

    # # Convert medical 3d volume data to image format
    # volume_generator = asus_nodule_volume_generator(data_path=merge_path, case_pids=case_pids)
    # medical_to_img.volumetric_data_preprocess(save_path=image_path, volume_generator=volume_generator)
    # # volume_generator = asus_nodule_volume_generator(data_path=merge_path)
    # # medical_to_img.volumetric_data_preprocess_KC(data_split, save_path=kc_image_path, volume_generator=volume_generator)

    # Build up coco-structure
    num_case = len(get_files(merge_path, recursive=False, get_dirs=True))
    cv_split_indices = get_cv_split(num_fold, num_case, shuffle)
        
    for fold in cv_split_indices:
        coco_split_path = os.path.join(coco_path, f'cv-{num_fold}', str(fold))
        if not os.path.isdir(coco_split_path):
            os.makedirs(coco_split_path)

        split_indices = cv_split_indices[fold]
        build_tmh_nodule_coco(
            data_path=image_path, save_path=coco_split_path, split_indices=split_indices, 
            cat_ids=cat_ids, area_threshold=area_threshold)


def build_parameters(config_path):
    cfg = load_config(config_path, dict_as_member=True)
    data_root = cfg.PATH.DATA_ROOT
    save_root = cfg.PATH.SAVE_ROOT
    task_name = cfg.TASK_NAME
    cat_ids = cfg.CATEGORY_ID[task_name]
    area_threshold = cfg.AREA_THRESHOLD
    data_split = cfg.SPLIT
    num_fold = cfg.CROSS_VALID_FOLD
    shuffle = True
    case_pids = None

    raw_path = os.path.join(data_root, 'raw')
    stats_path = os.path.join(save_root, 'stats_path')
    merge_path = os.path.join(save_root, 'merge')
    image_path = os.path.join(save_root, 'image')
    coco_path = os.path.join(save_root, 'coco', task_name)
    if task_name == 'Nodule_Detection':
        category = 'Nodule'
    elif task_name == 'Malignancy':
        category = 'Nodule'

    split_indices = {}
    for split_idx, split_name in data_split.items():
        if split_name in split_indices:
            split_indices[split_name].append(split_idx)
        else:
            split_indices[split_name] = [split_idx]

    data_parameters = {
        'data_root': data_root,
        'raw_path': raw_path,
        'cat_ids': cat_ids,
        'area_threshold': area_threshold,
        'stats_path': stats_path,
        'merge_path': merge_path,
        'image_path': image_path,
        'coco_path': coco_path,
        'category': category,
        'num_fold': num_fold,
        'shuffle': shuffle,
        'case_pids': case_pids,
    }
    return data_parameters


def remove_1m0045_noise(data_path):
    # load data
    itkimage = sitk.ReadImage(data_path)
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # remove noise
    ct_scan = cc3d.connected_components(ct_scan, 26)
    nodule_ids = np.unique(ct_scan)[1:]
    if nodule_ids.size > 1:
        nodule_sizes = {}
        for idx in nodule_ids:
            nodule_sizes[np.sum(ct_scan==idx)] = idx
        min_nodule_id = nodule_sizes[min(nodule_sizes.values())]
        new_ct_scan = np.where(ct_scan==min_nodule_id, 0, ct_scan)
        print(np.unique(new_ct_scan))

        # save data
        new_itk = modify_array_in_itk(itkimage, new_ct_scan)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(data_path)
        writer.Execute(new_itk)
        print('-- 1m0045 successfully modified')    
    else:
        print('-- 1m0045 has been modified')


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


def main():
    data_preprocess(CONFIG_PATH)


if __name__ == '__main__':
    main()
    # remove_1m0045_noise()
    
        