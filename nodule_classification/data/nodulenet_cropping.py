import os
import numpy as np
import cc3d
import nrrd
import matplotlib.pyplot as plt
import pandas as pd

from data.data_utils import get_files
from dataset_conversion.crop_data_utils import crop_volume, is_rich_context

# TODO: this mapping is for temporalily using, not a good solution.
def get_pid_tmh_mapping():
    f = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\merge'
    dir_list = [name for name in os.listdir(f) if os.path.isdir(os.path.join(f, name))]
    mapping = {}
    for _dir in dir_list:
        file_path = os.path.join(f, _dir, 'raw')
        folder_list = [name for name in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, name))]
        pid = folder_list[0][:-4]
        mapping[pid] = _dir
    return mapping


def nodulenet_data_cropping(data_root, save_dir, crop_range, context_threshold=0.5):
    """
    This is a small function to get the cropping result of nodulenet preprocessing
    dataset.
    """
    img_list = get_files(data_root, '_clean.nrrd')
    mask_list = get_files(data_root, '_mask.nrrd')
    idx = 0
    row_list = []
    # img_list = img_list[:5]
    # TODO: should be 'pid', 'tmh_name' not 'seriesuid', 'pid'
    pd_column = ['seriesuid', 'pid', 'crop_idx', 'center_i', 'center_r', 'center_c', 
                 'category', 'path', 'malignancy']
    pid_to_tmh_name = get_pid_tmh_mapping()
    for f_idx, (img_path, mask_path) in enumerate(zip(img_list, mask_list), 1):
        pid = os.path.split(img_path)[1].split('_')[0]
        tmh_name = pid_to_tmh_name[pid]
        img, _ = nrrd.read(img_path)
        mask, _ = nrrd.read(mask_path)
        mask_cat = cc3d.connected_components(mask, connectivity=26)
        center_pool = []
        print(f'{f_idx}/{len(img_list)} Processing {pid} {label}')
        for crop_idx, label in enumerate(np.unique(mask_cat)[1:]):
            z_range, y_range, x_range = np.where(mask_cat==label)
            z_cent = np.mean(z_range, dtype=np.int32)
            y_cent = np.mean(y_range, dtype=np.int32)
            x_cent = np.mean(x_range, dtype=np.int32)
            crop_center = {'index': z_cent, 'row': y_cent, 'column': x_cent}

            # positive
            crop_img = crop_volume(img, crop_range, crop_center)
            crop_mask = crop_volume(mask, crop_range, crop_center)
            file_name = f'{pid}_{idx:03d}.npy'
            idx += 1
            p_img_save_dir = os.path.join(save_dir, 'positive', 'Image')
            p_mask_save_dir = os.path.join(save_dir, 'positive', 'Mask')
            # plt.imshow(crop_img[16], 'gray')
            # plt.imshow(crop_mask[16], alpha=0.2)
            # plt.show()
            os.makedirs(p_img_save_dir, exist_ok=True)
            os.makedirs(p_mask_save_dir, exist_ok=True)
            np.save(os.path.join(p_img_save_dir, file_name), crop_img)
            np.save(os.path.join(p_mask_save_dir, file_name), crop_mask)
            if np.max(crop_mask) == 1:
                malignancy = 'benign'
            elif np.max(crop_mask) == 2:
                malignancy = 'malignant'
            p_df = pd.DataFrame([
                tmh_name, pid, crop_idx, z_cent, y_cent, x_cent, 
                'positive', os.path.join('positive', 'Image', file_name), malignancy],
                # columns=pd_column
            )
            row_list.append(p_df.T)

            # find a negative
            while 1:
                low = np.array([
                    crop_range['index']//2,
                    crop_range['row']//2,
                    crop_range['column']//2
                ])
                high = img.shape - low
                crop_center = np.random.randint(low, high)

                used = False
                for used_center in center_pool:
                    if np.all(crop_center==used_center):
                        used = True
                if used:
                    continue
                else:
                    center_pool.append(crop_center)

                z_cent, y_cent, x_cent = crop_center
                crop_center = {'index': z_cent, 'row': y_cent, 'column': x_cent}

                crop_mask = crop_volume(mask, crop_range, crop_center)
                crop_img = crop_volume(img, crop_range, crop_center)
                if not np.sum(crop_mask) and \
                    is_rich_context(crop_img, context_threshold, False):
                    n_img_save_dir = os.path.join(save_dir, 'negative', 'Image')
                    n_mask_save_dir = os.path.join(save_dir, 'negative', 'Mask')
                    os.makedirs(n_img_save_dir, exist_ok=True)
                    os.makedirs(n_mask_save_dir, exist_ok=True)
                    np.save(os.path.join(n_img_save_dir, file_name), crop_img)
                    np.save(os.path.join(n_mask_save_dir, file_name), crop_mask)
                    malignancy = 'null'
                    n_df = pd.DataFrame([
                        tmh_name, pid, crop_idx, z_cent, y_cent, x_cent, 
                        'negative', os.path.join('negative', 'Image', file_name), malignancy
                        ], 
                        # columns=pd_column
                    )
                    row_list.append(n_df.T)
                    break
    df = pd.concat(row_list, axis=0, ignore_index=True)
    df.columns = pd_column
    df.to_csv(os.path.join(save_dir, 'data_samples.csv'))


if __name__ == '__main__':
    data_root = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\nodulenet\preprocess'
    save_dir = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\nodulenet\crop'
    crop_range = {'index': 32, 'row': 64, 'column': 64}
    nodulenet_data_cropping(data_root, save_dir, crop_range)