import os
import cv2
import numpy as np
import pylidc as pl
import matplotlib.pyplot as plt
from pylidc.utils import consensus
import warnings
from utils import calculate_malignancy, raw_preprocess, mask_preprocess, compare_result, compare_result_enlarge
from volume_generator import luna16_volume_generator, asus_nodule_volume_generator
warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm

from modules.data import dataset_utils





def volumetric_data_preprocess_KC(data_path, save_path, vol_generator_func):
    volume_generator = vol_generator_func(data_path)
    for split in ['train', 'valid', 'test']:
        save_sub_dir = os.path.join(save_path, 'raw2')
        if not os.path.isdir(os.path.join(save_sub_dir, split, 'Image')):
            os.makedirs(os.path.join(save_sub_dir, split, 'Image'))
        if not os.path.isdir(os.path.join(save_sub_dir, split, 'Mask')):
            os.makedirs(os.path.join(save_sub_dir, split, 'Mask'))

    for vol_idx, (vol, mask_vol, infos) in enumerate(volume_generator):
        pid, scan_idx, subset = infos['pid'], infos['scan_idx'], infos['subset']
        # Create saving directry
        
        for img_idx in range(vol.shape[0]):
            if img_idx%50 == 0:
                print(f'Patient {pid} Scan {scan_idx} slice {img_idx}')
            img = vol[img_idx]
            mask = mask_vol[img_idx]
            if np.sum(mask):
                if subset in ['subset8', 'subset9']:
                    split = 'test'
                elif subset == 'subset7':
                    split = 'valid'
                else:
                    split = 'train'

                cv2.imwrite(os.path.join(save_sub_dir, split, 'Image', f'{pid}_{img_idx:04d}.png'), img)
                cv2.imwrite(os.path.join(save_sub_dir, split, 'Mask', f'{pid}_{img_idx:04d}.png'), mask)
    print('Preprocess complete!')


def volumetric_data_preprocess(data_path, save_path, vol_generator_func):
    volume_generator = vol_generator_func(data_path)
    for vol_idx, (vol, mask_vol, infos) in enumerate(volume_generator):
        pid, scan_idx, subset = infos['pid'], infos['scan_idx'], infos['subset']
        # Create saving directry
        save_sub_dir = os.path.join(save_path, 'raw', subset) if subset else os.path.join(save_path, 'raw') 
        if not os.path.isdir(os.path.join(save_sub_dir, 'Image', pid)):
            os.makedirs(os.path.join(save_sub_dir, 'Image', pid))
        if not os.path.isdir(os.path.join(save_sub_dir, 'Mask', pid)):
            os.makedirs(os.path.join(save_sub_dir, 'Mask', pid))
        if not os.path.isdir(os.path.join(save_sub_dir, 'Mask_show', pid)):
            os.makedirs(os.path.join(save_sub_dir, 'Mask_show', pid))
        
        for img_idx in range(vol.shape[0]):
            if img_idx%50 == 0:
                print(f'Patient {pid} Scan {scan_idx} slice {img_idx}')
            img = vol[img_idx]
            mask = mask_vol[img_idx]
            if np.sum(mask):
                mask_show = np.where(mask>0, 255, 0)
                fig, ax = plt.subplots(1, 1)
                ax.imshow(img, 'gray')
                ax.imshow(mask_show, alpha=0.2)
                fig.savefig(os.path.join(save_sub_dir, 'Mask_show', pid, f'{pid}_{img_idx:04d}.png'))
                # cv2.imwrite(os.path.join(save_sub_dir, 'Mask_show', pid, f'{pid}_{img_idx:04d}.png'), mask_show)

            cv2.imwrite(os.path.join(save_sub_dir, 'Image', pid, f'{pid}_{img_idx:04d}.png'), img)
            cv2.imwrite(os.path.join(save_sub_dir, 'Mask', pid, f'{pid}_{img_idx:04d}.png'), mask)
    print('Preprocess complete!')
            

def luna16_volume_preprocess():
    src = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data'
    dst = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess'
    volumetric_data_preprocess(data_path=src, 
                               save_path=dst, 
                               vol_generator_func=luna16_volume_generator.Build_DLP_luna16_volume_generator)


def luna16_volume_preprocess_round():
    src = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data'
    dst = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess-round'
    volumetric_data_preprocess(data_path=src, 
                               save_path=dst, 
                               vol_generator_func=luna16_volume_generator.Build_Round_luna16_volume_generator)


def asus_benign_volume_preprocess():
    src = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\benign'
    dst = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\benign'
    volumetric_data_preprocess(data_path=src, save_path=dst, vol_generator_func=asus_nodule_volume_generator)


def asus_malignant_volume_preprocess():
    src = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant'
    dst = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\malignant'
    volumetric_data_preprocess(data_path=src, save_path=dst, vol_generator_func=asus_nodule_volume_generator)

    
if __name__ == '__main__':
    # luna16_volume_preprocess()
    # luna16_volume_preprocess_round()
    asus_benign_volume_preprocess()
    # asus_malignant_volume_preprocess()
    pass