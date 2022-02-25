import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylidc as pl
from pylidc.utils import consensus
import warnings
from utils.utils import calculate_malignancy, raw_preprocess, mask_preprocess, compare_result, compare_result_enlarge
from utils.volume_generator import luna16_volume_generator, asus_nodule_volume_generator
warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm

from modules.data import dataset_utils


# def get_data_split(dataset_name, subset, scan_idx):
#     if dataset_name == 'LUNA16':
#         if subset in ['subset8', 'subset9']:
#             split = 'test'
#         elif subset == 'subset7':
#             split = 'valid'
#         else:
#             split = 'train'
#     elif dataset_name == 'ASUS-Benign':
#         if scan_idx in list(range(1,26)):
#             split = 'train'
#         elif scan_idx in list(range(28, 36)):
#             split = 'test'
#         else:
#             split = 'valid'
#     elif dataset_name == 'ASUS-Malignant':
#         if scan_idx in list(range(1,41)):
#             split = 'train'
#         elif scan_idx in list(range(46, 58)):
#             split = 'test'
#         else:
#             split = 'valid'
#     return split


def volumetric_data_preprocess(data_path, save_path, vol_generator_func):
    volume_generator = vol_generator_func(data_path)
    for vol_idx, (_, vol, mask_vol, infos) in enumerate(volume_generator):
        pid, scan_idx, subset = infos['pid'], infos['scan_idx'], infos['subset']
        print(f'Patient {pid} Scan {scan_idx}')
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
    dst = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\raw'
    # volumetric_data_preprocess(data_path=src, save_path=dst, vol_generator_func=luna16_volume_generator.Build_DLP_luna16_volume_generator)
    volumetric_data_preprocess(dataset_name='LUNA16', 
                               data_path=src, 
                               save_path=dst.replace('raw', 'LUNA16_preprocess2'), 
                               vol_generator_func=luna16_volume_generator.Build_DLP_luna16_volume_generator)


def luna16_volume_preprocess_round():
    src = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data'
    dst = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess-round\raw'
    # volumetric_data_preprocess(data_path=src, save_path=dst, vol_generator_func=luna16_volume_generator.Build_Round_luna16_volume_generator)
    volumetric_data_preprocess(dataset_name='LUNA16', 
                               data_path=src, 
                               save_path=dst.replace('raw', 'LUNA16_preprocess'), 
                               vol_generator_func=luna16_volume_generator.Build_Round_luna16_volume_generator)


def asus_benign_volume_preprocess():
    src = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\benign_merge'
    dst = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\benign\raw_merge'
    volumetric_data_preprocess(data_path=src, save_path=dst, vol_generator_func=asus_nodule_volume_generator)
    # volumetric_data_preprocess_KC(dataset_name='ASUS-Benign', 
    #                               data_path=src, 
    #                               save_path=dst.replace('raw', 'ASUS_benign_preprocess2'), 
    #                               vol_generator_func=asus_nodule_volume_generator)


def asus_malignant_volume_preprocess():
    src = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant_merge'
    dst = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\malignant\raw_merge'
    volumetric_data_preprocess(data_path=src, save_path=dst, vol_generator_func=asus_nodule_volume_generator)
    # volumetric_data_preprocess_KC(dataset_name='ASUS-Malignant', 
    #                               data_path=src, 
    #                               save_path=dst.replace('raw', 'ASUS_malignant_preprocess2'), 
    #                               vol_generator_func=asus_nodule_volume_generator)


def main():
    # luna16_volume_preprocess()
    # luna16_volume_preprocess_round()
    asus_benign_volume_preprocess()
    asus_malignant_volume_preprocess()


if __name__ == '__main__':
    main()
    pass