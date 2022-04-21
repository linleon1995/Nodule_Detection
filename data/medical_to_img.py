import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylidc as pl
from pylidc.utils import consensus
import warnings
# from utils.utils import calculate_malignancy, raw_preprocess, mask_preprocess, compare_result, compare_result_enlarge
from data.volume_generator import luna16_volume_generator, asus_nodule_volume_generator
warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm
import site_path


        
def volumetric_data_preprocess_KC(data_split, save_path, volume_generator):
    make_dir = lambda path: os.makedirs(path) if not os.path.isdir(path) else None

    for keep_nodules in ['all', 'nodule']:
        for split in ['train', 'valid', 'test']:
            make_dir(os.path.join(save_path, keep_nodules, split))

    all_save_path = os.path.join(save_path, 'all')
    nodule_save_path = os.path.join(save_path, 'nodule')
    
    for vol_idx, (_, vol, mask_vol, infos) in enumerate(volume_generator):
        pid, scan_idx, subset = infos['pid'], infos['scan_idx'], infos['subset']
        print(f'Patient {pid} Scan {vol_idx}')
        subset = '' if subset is None else subset
        
        split = data_split[vol_idx]
        for img_idx in range(vol.shape[0]):
            img = vol[img_idx][...,0]
            mask = mask_vol[img_idx]

            make_dir(os.path.join(all_save_path, split, subset, pid, 'Image')) 
            make_dir(os.path.join(all_save_path, split, subset, pid, 'Mask'))
            cv2.imwrite(os.path.join(all_save_path, split, subset, pid, 'Image', f'{pid}_{img_idx:04d}.png'), img)
            cv2.imwrite(os.path.join(all_save_path, split, subset, pid,  'Mask', f'{pid}_{img_idx:04d}.png'), mask)

            if np.sum(mask):
                make_dir(os.path.join(nodule_save_path, split, subset, pid, 'Image')) 
                make_dir(os.path.join(nodule_save_path, split, subset, pid, 'Mask'))
                cv2.imwrite(os.path.join(nodule_save_path, split, subset, pid, 'Image', f'{pid}_{img_idx:04d}.png'), img)
                cv2.imwrite(os.path.join(nodule_save_path, split, subset, pid,  'Mask', f'{pid}_{img_idx:04d}.png'), mask)
    print('Complete converting process of mhd to image (KC)!')


def volumetric_data_preprocess(save_path, volume_generator):
    fig, ax = plt.subplots(1, 1, dpi=300)
    for vol_idx, (_, vol, mask_vol, infos) in enumerate(volume_generator):
        pid, scan_idx, subset = infos['pid'], infos['scan_idx'], infos['subset']
        print(f'Patient {pid} Scan {vol_idx}')

        # Create saving directry
        save_sub_dir = os.path.join(save_path, subset) if subset else save_path
        if not os.path.isdir(os.path.join(save_sub_dir, 'Image', pid)):
            os.makedirs(os.path.join(save_sub_dir, 'Image', pid))
        if not os.path.isdir(os.path.join(save_sub_dir, 'Mask', pid)):
            os.makedirs(os.path.join(save_sub_dir, 'Mask', pid))
        if not os.path.isdir(os.path.join(save_sub_dir, 'Mask_show', pid)):
            os.makedirs(os.path.join(save_sub_dir, 'Mask_show', pid))
        
        for img_idx in range(vol.shape[0]):
            img = vol[img_idx]
            mask = mask_vol[img_idx]
            if np.sum(mask):
                mask_show = np.where(mask>0, 255, 0)
                
                ax.cla()
                ax.imshow(img, 'gray')
                ax.imshow(mask_show, alpha=0.2)
                fig.savefig(os.path.join(save_sub_dir, 'Mask_show', pid, f'{pid}_{img_idx:04d}.png'))
                # cv2.imwrite(os.path.join(save_sub_dir, 'Mask_show', pid, f'{pid}_{img_idx:04d}.png'), mask_show)

            cv2.imwrite(os.path.join(save_sub_dir, 'Image', pid, f'{pid}_{img_idx:04d}.png'), img)
            cv2.imwrite(os.path.join(save_sub_dir, 'Mask', pid, f'{pid}_{img_idx:04d}.png'), mask)
    print('Complete converting process of mhd to image!')
            
