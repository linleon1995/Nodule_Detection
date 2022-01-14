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
                cv2.imwrite(os.path.join(save_sub_dir, 'Mask_show', pid, f'{pid}_{img_idx:04d}.png'), mask_show)

            cv2.imwrite(os.path.join(save_sub_dir, 'Image', pid, f'{pid}_{img_idx:04d}.png'), img)
            cv2.imwrite(os.path.join(save_sub_dir, 'Mask', pid, f'{pid}_{img_idx:04d}.png'), mask)
    print('Preprocess complete!')
            


def lidc_preprocess(data_path, save_path, clevel=0.2, padding=512):
    case_list = dataset_utils.get_files(data_path, keys=[], return_fullpath=False, sort=True, recursive=False, get_dirs=True)
    # case_list = case_list[209:820]
    case_list = case_list[730:]
    # case_list = case_list[820:830]
    for pid in tqdm(case_list):
        scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid)
        num_scan_in_one_patient = scans.count()
        print(f'{pid} has {num_scan_in_one_patient} scan')
        scan_list = scans.all()
        num_pid = pid.split('-')[-1]
        for scan_idx, scan in enumerate(scan_list):
            # scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            # +++
            if scan is None:
                print(scan)
            nodules_annotation = scan.cluster_annotations()
            vol = scan.to_volume()
            
            print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid, vol.shape, len(nodules_annotation)))

            # Make directory
            save_vol_path = os.path.join(save_path, pid)
            full_vol_npy = os.path.join(save_vol_path, 'Image', 'full', 'vol', 'npy')
            full_img_npy = os.path.join(save_vol_path, 'Image', 'full', 'img', 'npy')
            full_img_png = os.path.join(save_vol_path, 'Image', 'full', 'img', 'png')
            full_vol_mask_npy = os.path.join(save_vol_path, 'Mask', 'vol', 'npy')
            full_mask_npy = os.path.join(save_vol_path, 'Mask', 'img', 'npy')
            full_mask_png = os.path.join(save_vol_path, 'Mask', 'img', 'png')
            full_mask_vis = os.path.join(save_vol_path, 'Mask', 'img', 'vis')
            lung_vol_npy = full_vol_npy.replace('full', 'lung')
            lung_img_npy = full_img_npy.replace('full', 'lung')
            lung_img_png = full_img_png.replace('full', 'lung')

            dir_list = []
            dir_list.append(full_vol_npy)
            dir_list.append(full_img_npy)
            dir_list.append(full_img_png)
            dir_list.append(lung_vol_npy)
            dir_list.append(lung_img_npy)
            dir_list.append(lung_img_png)
            dir_list.append(full_vol_mask_npy)
            dir_list.append(full_mask_npy)
            dir_list.append(full_mask_png)
            dir_list.append(full_mask_vis)
            for _dir in dir_list:
                if not os.path.isdir(_dir):
                    os.makedirs(_dir)

            # Patients with nodules
            
            masks_vol = np.zeros_like(vol) # for all masks
            for nodule_idx, nodule in enumerate(nodules_annotation):
                one_mask_vol = np.zeros_like(vol)
                # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
                # This current for loop iterates over total number of nodules in a single patient
                mask, cbbox, _ = consensus(nodule, clevel=clevel, pad=padding)
                # assert np.shape(vol) == np.shape(mask), f'The input image shape {np.shape(vol)} and mask shape {np.shape(mask)} must be the same.'
                # Regard Ambiuious as malignant
                malignancy, cancer_label = calculate_malignancy(nodule)
                one_mask_vol[cbbox] = mask
                masks_vol += malignancy*one_mask_vol

                # if nodule_idx == 2:
                #     img1 = one_mask_vol[...,218]
                # if nodule_idx == 3:
                #     img2 = one_mask_vol[...,218]
                
            #     # plt.imshow(one_mask_vol[...,218])
            #     # plt.show()
            # a = np.where(masks_vol==6)
            # # plt.imshow(masks_vol[...,218])
            # print(np.sum(img1*img2))
            # # plt.show()

            for i in range(masks_vol.shape[2]):
                print(np.sum(masks_vol[...,i]))
            assert np.max(masks_vol) <= 5, f'The nodule malignancy {np.max(masks_vol)} higher than 5'

            vol_lung = np.zeros_like(vol)
            for img_idx in range(vol.shape[2]):
                # print(f'Patient {pid} Scan {scan_idx} slice {img_idx}')
                img = vol[...,img_idx]
                img = np.int16(img)
                lung_img = img.copy()
                # lung_img = segment_lung(lung_img)
                # def process(img):
                #     img[img==-0] = 0
                #     if np.min(img) == np.max(img):
                #         img = np.zeros_like(img)
                #     else:
                #         img = 255*((img-np.min(img))/(np.max(img)-np.min(img)))
                #     return np.uint8(img)

                # img = process(img)
                # lung_img = process(lung_img)
                lung_img = raw_preprocess(lung_img, lung_segment=True, change_channel=False)
                vol[...,img_idx] = img
                vol_lung[...,img_idx] = lung_img

                img_name = f'{num_pid}-Scan{scan_idx}-Image{img_idx:03d}'
                cv2.imwrite(os.path.join(full_img_png, f'{img_name}.png'), img)
                cv2.imwrite(os.path.join(lung_img_png, f'{img_name}.png'), lung_img)
                # np.save(os.path.join(full_img_npy, f'{img_name}.npy'), lung_img)
                # np.save(os.path.join(lung_img_npy, f'{img_name}.npy'), img)

                mask_name = f'{num_pid}-Scan{scan_idx}-Mask{img_idx:03d}'
                cv2.imwrite(os.path.join(full_mask_png, f'{mask_name}.png'), masks_vol[...,img_idx])
                if np.sum(masks_vol[...,img_idx]) > 0:
                    cv2.imwrite(os.path.join(full_mask_vis, f'{mask_name}.png'), masks_vol[...,img_idx]*127)
                # np.save(os.path.join(full_mask_npy, f'{mask_name}.npy'), masks_vol[...,img_idx])

            vol_name = f'{num_pid}-Scan{scan_idx}'
            np.save(os.path.join(full_vol_npy, f'{vol_name}.npy'), np.int16(vol))
            np.save(os.path.join(lung_vol_npy, f'{vol_name}.npy'), np.int16(vol_lung))
            np.save(os.path.join(full_vol_mask_npy, f'{vol_name}.npy'), np.int16(masks_vol))


def luna16_volume_preprocess():
    src = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data'
    dst = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess'
    volumetric_data_preprocess_KC(data_path=src, 
                               save_path=dst, 
                               vol_generator_func=luna16_volume_generator.Build_DLP_luna16_volume_generator)


def luna16_volume_preprocess_round():
    src = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data'
    dst = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess-round'
    volumetric_data_preprocess(data_path=src, 
                               save_path=dst, 
                               vol_generator_func=luna16_volume_generator.Build_Round_luna16_volume_generator)


def asus_nodule_volume_preprocess():
    src = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant'
    dst = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\malignant'
    volumetric_data_preprocess(data_path=src, save_path=dst, vol_generator_func=asus_nodule_volume_generator)


if __name__ == '__main__':
    luna16_volume_preprocess()
    # asus_nodule_volume_preprocess()
    pass