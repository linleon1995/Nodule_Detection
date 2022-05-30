import os
import numpy as np
import cc3d
import SimpleITK as sitk
import pandas as pd

from dataset_conversion import medical_to_img
from dataset_conversion.build_coco import build_tmh_nodule_coco
from dataset_conversion.TMH import tmh_data_merge
from dataset_conversion.data_analysis import get_nodule_diameter
# from dataset_conversion.data_analysis import TMH_nodule_base_check
from data.volume_generator import asus_nodule_volume_generator
from data.data_utils import modify_array_in_itk, get_files, load_itk
from postprocessing.lung_mask_filtering import segment_lung
from utils.train_utils import set_deterministic

from utils.configuration import load_config
import matplotlib.pyplot as plt
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# DATASET_NAME = 'TMH-Nodule'
CONFIG_PATH = 'dataset_conversion/config/TMH-Nodule.yml'


def data_preprocess(dataset_name):
    dataset_parameter = build_parameters(dataset_name)

    raw_path = dataset_parameter['raw_path']
    stats_path = dataset_parameter['stats_path']
    merge_path = dataset_parameter['merge_path']
    image_path = dataset_parameter['image_path']
    coco_path = dataset_parameter['coco_path']
    cat_ids = dataset_parameter['cat_ids']
    area_threshold = dataset_parameter['area_threshold']
    # category = dataset_parameter['category']
    num_fold = dataset_parameter['num_fold']
    shuffle = dataset_parameter['shuffle']
    case_pids = dataset_parameter['case_pids']
    n_class = dataset_parameter['n_class']
    height = dataset_parameter['height']
    width = dataset_parameter['width']
    kc_image_path = image_path.replace('image', 'kc_image')
    for path in [merge_path, image_path, kc_image_path, stats_path]:
        os.makedirs(path, exist_ok=True)
    
    # # TMH base check
    # raw_paths = get_files(raw_path, recursive=False, get_dirs=True)
    # volume_generator = asus_nodule_volume_generator(data_path=raw_paths, case_pids=case_pids)
    # TMH_nodule_base_check(volume_generator, save_path=stats_path)

    # Merge mhd data
    merge_mapping = tmh_data_merge.TMH_merging_check(raw_path, merge_path)
    tmh_data_merge.merge_data(merge_mapping, raw_path, merge_path, filekey='TMH')

    # # Convert medical 3d volume data to image format
    # volume_generator = asus_nodule_volume_generator(data_path=merge_path, case_pids=case_pids)
    # medical_to_img.volumetric_data_preprocess(
    #     save_path=image_path, volume_generator=volume_generator, n_class=n_class)
    # # volume_generator = asus_nodule_volume_generator(data_path=merge_path)
    # # medical_to_img.volumetric_data_preprocess_KC(data_split, save_path=kc_image_path, volume_generator=volume_generator)

    # # Build up coco-structure
    # for task_name in cat_ids:
    #     task_cat_ids = cat_ids[task_name]
    #     task_coco_path = os.path.join(coco_path, task_name)

    #     num_case = len(get_files(merge_path, recursive=False, get_dirs=True))
    #     cv_split_indices = get_cv_split(num_fold, num_case, shuffle)
    #     for fold in cv_split_indices:
    #         coco_split_path = os.path.join(task_coco_path, f'cv-{num_fold}', str(fold))
    #         os.makedirs(coco_split_path, exist_ok=True)

    #         split_indices = cv_split_indices[fold]
    #         build_tmh_nodule_coco(
    #             data_path=image_path, save_path=coco_split_path, split_indices=split_indices, 
    #             cat_ids=task_cat_ids, area_threshold=area_threshold, height=height, width=width
    #         )


def build_parameters(config_path):
    cfg = load_config(config_path, dict_as_member=True)
    raw_path = cfg.PATH.DATA_ROOT
    save_root = cfg.PATH.SAVE_ROOT
    # task_name = cfg.TASK_NAME
    cat_ids = cfg.CATEGORY_ID
    area_threshold = cfg.AREA_THRESHOLD
    # data_split = cfg.SPLIT
    num_fold = cfg.CROSS_VALID_FOLD
    shuffle = True
    case_pids = None
    set_deterministic(cfg.SEED)

    stats_path = os.path.join(save_root, 'stats_path')
    merge_path = os.path.join(save_root, 'merge')
    image_path = os.path.join(save_root, 'image')
    coco_path = os.path.join(save_root, 'coco')
    # if task_name == 'Nodule_Detection':
    #     category = 'Nodule'
    # elif task_name == 'Malignancy':
    #     category = 'Nodule'

    # split_indices = {}
    # for split_idx, split_name in data_split.items():
    #     if split_name in split_indices:
    #         split_indices[split_name].append(split_idx)
    #     else:
    #         split_indices[split_name] = [split_idx]

    data_parameters = {
        'raw_path': raw_path,
        'cat_ids': cat_ids,
        'area_threshold': area_threshold,
        'stats_path': stats_path,
        'merge_path': merge_path,
        'image_path': image_path,
        'coco_path': coco_path,
        # 'category': category,
        'num_fold': num_fold,
        'shuffle': shuffle,
        'case_pids': case_pids,
        'n_class': cfg.N_CLASS,
        'height': cfg.HEIGHT,
        'width': cfg.WIDTH,
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



def TMH_nodule_base_check(volume_generator, save_path=None):
    total_size, total_infos, total_diameters = {}, {}, []
    check_path = os.path.join(save_path, 'checked')
    for vol_idx, (raw_vol, vol, mask_vol, infos) in enumerate(volume_generator):
        cat_vol = cc3d.connected_components(mask_vol, connectivity=26) # category volume
        nodule_ids = np.unique(cat_vol)[1:]
        pid = infos['pid']
        origin, spacing, direction = infos['origin'], infos['spacing'], infos['direction']

        for n_id in nodule_ids:
            nodule_vol = np.int32(cat_vol==n_id)
            nodule_size = np.sum(nodule_vol)
            nodule_id = f'{pid}_{n_id:03d}'
            total_size[nodule_id] = nodule_size

            zs, ys, xs = np.where(nodule_vol)
            unique_zs = np.unique(zs)
            min_z, max_z = np.min(unique_zs), np.max(unique_zs)

            nodule_diameter = get_nodule_diameter(nodule_vol, origin, spacing, direction)
            total_diameters.append(nodule_diameter)

            if vol_idx == 0 and n_id == 1:
                total_infos['seriesuid'] = [pid]
                total_infos['nodule_id'] = [n_id]
                total_infos['nodule_start'] = [min_z]
                total_infos['nodule_end'] = [max_z]
                total_infos['slice_num'] = [max_z-min_z+1]
                total_infos['nodule_size'] = [nodule_size]
                total_infos['nodule_diameter'] = [nodule_diameter]
            else:
                total_infos['seriesuid'].append(pid)
                total_infos['nodule_id'].append(n_id)
                total_infos['nodule_start'].append(min_z)
                total_infos['nodule_end'].append(max_z)
                total_infos['slice_num'].append(max_z-min_z+1)
                total_infos['nodule_size'].append(nodule_size)
                total_infos['nodule_diameter'].append(nodule_diameter)

            # for z_idx in unique_zs:
            #     img_save_path = os.path.join(check_path, pid)
            #     os.makedirs(img_save_path, exist_ok=True)
            #     show_mask_base(vol[z_idx], mask_vol[z_idx], save_path=os.path.join(img_save_path, f'img_{z_idx:03d}'))
        
            print(f'{vol_idx} Nodule {nodule_id}  size {nodule_size} pixels   diameter {nodule_diameter} mm')


    print(20*'-')
    df = pd.DataFrame(total_infos)
    os.makedirs(check_path, exist_ok=True)
    df.to_csv(os.path.join(check_path, 'TMH_nodule_check.csv'), index=False)
    nodule_sizes = list(total_size.values())
    max_size_nodule = list(total_size.keys())[list(total_size.values()).index(max(nodule_sizes))]
    min_size_nodule = list(total_size.keys())[list(total_size.values()).index(min(nodule_sizes))]
    print(f'Nodule number {len(nodule_sizes)}')
    print(f'Max size: {max_size_nodule} {max(nodule_sizes)}')
    print(f'Min size: {min_size_nodule} {min(nodule_sizes)}')
    print(f'Mean size: {sum(nodule_sizes)/len(nodule_sizes)}')

    print(f'Max diameter: {max(total_diameters)}')
    print(f'Min diameter: {min(total_diameters)}')
    print('\n')



def convert_lung_mask():
    img_path = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\merge'
    save_dir = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\nodulenet'
    pid_list = get_files(img_path, get_dirs=True, return_fullpath=False, recursive=False)
    f_list = get_files(img_path, 'mhd')

    img_list = []
    mask_list = []
    for f in f_list:
        if 'raw' in f:
            img_list.append(f)
        if 'mask' in f:
            mask_list.append(f)
    img_list.sort()
    mask_list.sort()
    pid_list.sort()

    for idx, (img_path, pid) in enumerate(zip(img_list, pid_list)):
        print(idx)
        ct, _, _, _ = load_itk(img_path)
        lung_mask_vol = np.zeros_like(ct)
        case_dir = os.path.join(save_dir, pid)
        case_img_dir = os.path.join(case_dir, 'img')
        os.makedirs(case_img_dir, exist_ok=True)

        for slice, img in enumerate(ct):
            lung_mask = segment_lung(img)
            lung_mask_vol[slice] = lung_mask
            plt.imshow(img*lung_mask)
            plt.savefig(os.path.join(case_img_dir, f'lung-{idx}-{slice}.png'))
            plt.close()
    
        # seg_img = ct * lung_mask_vol
        np.save(os.path.join(case_dir, f'{pid}.npy'), lung_mask_vol)

            



def main():
    data_preprocess(CONFIG_PATH)


if __name__ == '__main__':
    # main()


    # convert_lung_mask()


    # f = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\nodulenet\TMH0002\TMH0002.npy'
    # a = np.load(f)
    # for idx, img in enumerate(a):
    #     if np.sum(img):
    #         plt.imshow(img)
    #         plt.title(str(idx))
    #         plt.show()


    # remove_1m0045_noise()


    ff = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\nodulenet_lung_mask'
    f_list = get_files(ff, 'npy')
    for f in f_list:
        path, old_name = os.path.split(f)
        new_name = old_name.replace('.mhd', '')
        os.rename(f, os.path.join(path, new_name))
        
    
        