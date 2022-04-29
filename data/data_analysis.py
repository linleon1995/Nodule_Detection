from dis import dis
import numpy as np
import matplotlib.pyplot as plt
import cv2
import cc3d
import os
import pandas as pd
from py import process
from utils.vis import show_mask_base
from utils.utils import xyz2irc
from data.data_utils import get_files, load_itk


def TMH_merging_check():
    benign_root = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodule\TMH-Benign\raw'
    malignant_root = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodule\TMH-Malignant\raw'

    benign_paths = get_files(benign_root, 'mhd')
    malignant_paths = get_files(malignant_root, 'mhd')

    total_paths = benign_paths + malignant_paths

    # remove mask path
    for idx, path in enumerate(total_paths):
        if 'mask' in path:
            total_paths.remove(path)
    total_paths.pop(3)
    
    output_paths = total_paths.copy()
    merge_table = {}

    process_list =[]
    for idx, path in enumerate(total_paths):
        folder, filename = os.path.split(path)
        _, pid = os.path.split(os.path.split(folder)[0])
        if pid in process_list:
            continue
        else:
            process_list.append(pid)
        temp_paths = total_paths.copy()
        temp_paths.remove(path)
        same_list = [pid]
        # print(idx, pid)
        # total_paths.remove(path)
        for compare_path in temp_paths:
            compare_folder, compare_filename = os.path.split(compare_path)
            _, compare_pid = os.path.split(os.path.split(compare_folder)[0])
            if filename == compare_filename:
                same_name = True
            else:
                same_name = False

            vol, _, _, _ = load_itk(path)
            compare_vol, _, _, _ = load_itk(compare_path)
            same_value = False
            if vol.shape == compare_vol.shape:
                if (vol == compare_vol).all():
                    same_value = True

            if same_name or same_value:
                same_list.append(compare_pid)
                process_list.append(compare_pid)
                # temp_paths.remove(compare_path)
                # total_paths.remove(compare_path)
                print(f'pid {pid} compare_pid {compare_pid} name {same_name} value {same_value}')
        merge_table[idx] = same_list
        
        print(same_list)
    df = pd.DataFrame(merge_table)
    df.to_csv('merge_table.csv')
    print(3)

def build_nodule_metadata(volume):
    if np.sum(volume) == np.sum(np.zeros_like(volume)):
        return None

    nodule_category = np.unique(volume)
    nodule_category = np.delete(nodule_category, np.where(nodule_category==0))
    total_nodule_metadata = []
    for label in nodule_category:
        binary_mask = volume==label
        nodule_size = np.sum(binary_mask)
        zs, ys, xs = np.where(binary_mask)
        center = {'index': np.mean(zs), 'row': np.mean(ys), 'column': np.mean(xs)}
        nodule_metadata = {'Nodule_id': label,
                            'Nodule_size': nodule_size,
                            'Nodule_slice': (np.min(zs), np.max(zs)),
                            'Noudle_center': center}
        total_nodule_metadata.append(nodule_metadata)
    return total_nodule_metadata


def build_nodule_distribution(ax, x, y, s, color, label):
    sc = ax.scatter(x, y, s=s, alpha=0.5, color=color, label=label)
    ax.set_title('The size and space distribution of lung nodules')
    ax.set_xlabel('row')
    ax.set_ylabel('column')
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.legend()
    # ax.legend(*sc.legend_elements("sizes", num=4))
    return ax


def single_nodule_distribution(ax, volume_list, color, label):
    size_list = []
    x, y, = [], []
    for volume in volume_list:
        total_nodule_info = build_nodule_metadata(volume)
        print(total_nodule_info)

        for nodule_info in total_nodule_info:
            size_list.append(nodule_info['Nodule_size'])
            x.append(np.int32(nodule_info['Noudle_center']['column']))
            y.append(np.int32(nodule_info['Noudle_center']['row']))
    ax = build_nodule_distribution(ax, x, y, size_list, color, label)
    return ax


def multi_nodule_distribution(train_volumes, test_volumes):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\benign\raw\Image\1B004\1B004_0169.png'
    img = cv2.imread(path)
    ax.imshow(img)

    ax = single_nodule_distribution(ax, train_volumes, color='b', label='train')
    ax = single_nodule_distribution(ax, test_volumes, color='orange', label='test')

    fig.show()
    fig.savefig('lung.png')


def get_nodule_diameter(nodule_vol, origin_zyx, spacing_zyx, direction_zyx):
    zs, ys, xs = np.where(nodule_vol)
    total_dist = []
    for idx, (z, y, x) in enumerate(zip(zs, ys, xs)):
        dist = (z**2 + y**2 + x**2)**0.5
        total_dist.append(dist)
    min_dist = min(total_dist)
    max_dist = max(total_dist)
    min_nodule = total_dist.index(min_dist)
    max_nodule = total_dist.index(max_dist)
    min_point_zyx = np.array((zs[min_nodule], ys[min_nodule], xs[min_nodule]))
    max_point_zyx = np.array((zs[max_nodule], ys[max_nodule], xs[max_nodule]))
    min_point_irc = xyz2irc(min_point_zyx, origin_zyx, spacing_zyx, direction_zyx)
    max_point_irc = xyz2irc(max_point_zyx, origin_zyx, spacing_zyx, direction_zyx)

    nodule_diameter = (np.sum((min_point_irc - max_point_irc)**2))**0.5
    return nodule_diameter


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
