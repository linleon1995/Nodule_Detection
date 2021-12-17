import os
import cv2
import numpy as np
import site_path
import pylidc as pl
from pylidc.utils import consensus
from statistics import median_high
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter
from skimage import measure, morphology
from sklearn.cluster import KMeans
from tqdm import tqdm

from modules.data import dataset_utils



def calculate_malignancy(nodule):
    # Calculate the malignancy of a nodule with the annotations made by 4 doctors. Return median high of the annotated cancer, True or False label for cancer
    # if median high is above 3, we return a label True for cancer
    # if it is below 3, we return a label False for non-cancer
    # if it is 3, we return ambiguous
    list_of_malignancy =[]
    for annotation in nodule:
        list_of_malignancy.append(annotation.malignancy)

    malignancy = median_high(list_of_malignancy)
    if  malignancy > 3:
        return malignancy, True
    elif malignancy < 3:
        return malignancy, False
    else:
        return malignancy, 'Ambiguous'


def segment_lung(img):
    #function sourced from https://www.kaggle.com/c/data-science-bowl-2017#tutorial
    """
    This segments the Lung Image(Don't get confused with lung nodule segmentation)
    """
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    
    middle = img[100:400,100:400] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    #remove the underflow bins
    img[img==max]=mean
    img[img==min]=mean
    
    #apply median filter
    img= median_filter(img,size=3)
    #apply anistropic non-linear diffusion filter- This removes noise without blurring the nodule boundary
    img= anisotropic_diffusion(img)
    
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
    eroded = morphology.erosion(thresh_img,np.ones([4,4]))
    dilation = morphology.dilation(eroded,np.ones([10,10]))
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
            good_labels.append(prop.label)
    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0
    #
    #  The mask here is the mask for the lungs--not the nodes
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    # mask consists of 1 and 0. Thus by mutliplying with the orginial image, sections with 1 will remain
    return mask*img



def lidc_preprocess(path, save_path, clevel=0.5, padding=512):
    case_list = dataset_utils.get_files(path, keys=[], return_fullpath=False, sort=True, recursive=False, get_dirs=True)
    case_list = case_list[:10]
    # case_list = case_list[810:820]
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
            masks_vol = np.zeros_like(vol)
            for nodule_idx, nodule in enumerate(nodules_annotation):
                # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
                # This current for loop iterates over total number of nodules in a single patient
                mask, cbbox, masks = consensus(nodule, clevel=clevel, pad=padding)
                assert np.shape(vol) == np.shape(mask), 'The input image shape and mask shape should be the same.'
                # Regard Ambiuious as malignant
                malignancy, cancer_label = calculate_malignancy(nodule)
                if malignancy >= 3:
                    cancer_categories = 2
                else:
                    cancer_categories = 1
                masks_vol += cancer_categories*mask

            vol_lung = np.zeros_like(vol)
            for img_idx in range(vol.shape[2]):
                # print(f'Patient {pid} Scan {scan_idx} slice {img_idx}')
                img = vol[...,img_idx]
                lung_img = img.copy()
                lung_img = segment_lung(lung_img)
                def process(img):
                    img[img==-0] = 0
                    if np.min(img) == np.max(img):
                        img = np.zeros_like(img)
                    else:
                        img = 255*((img-np.min(img))/(np.max(img)-np.min(img)))
                    return np.uint8(img)

                img = process(img)
                lung_img = process(lung_img)
                vol[...,img_idx] = img
                vol_lung[...,img_idx] = lung_img

                img_name = f'{num_pid}-Scan{scan_idx}-Image{img_idx:03d}'
                cv2.imwrite(os.path.join(full_img_png, f'{img_name}.png'), img)
                cv2.imwrite(os.path.join(lung_img_png, f'{img_name}.png'), lung_img)
                np.save(os.path.join(full_img_npy, f'{img_name}.npy'), lung_img)
                np.save(os.path.join(lung_img_npy, f'{img_name}.npy'), img)

                mask_name = f'{num_pid}-Scan{scan_idx}-Mask{img_idx:03d}'
                cv2.imwrite(os.path.join(full_mask_png, f'{mask_name}.png'), masks_vol[...,img_idx])
                cv2.imwrite(os.path.join(full_mask_vis, f'{mask_name}.png'), masks_vol[...,img_idx]*127)
                np.save(os.path.join(full_mask_npy, f'{mask_name}.npy'), masks_vol[...,img_idx])

            vol_name = f'{num_pid}-Scan{scan_idx}'
            np.save(os.path.join(full_vol_npy, f'{vol_name}.npy'), vol)
            np.save(os.path.join(lung_vol_npy, f'{vol_name}.npy'), vol_lung)
            np.save(os.path.join(full_vol_mask_npy, f'{vol_name}.npy'), masks_vol)



def cv2_imshow(img):
    # pass
    cv2.imshow('My Image', img)
    cv2.imwrite('sample.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convert_npy_to_png(src, dst, src_format, dst_format):
    src_files = dataset_utils.get_files(src, src_format)
    if not os.path.isdir(dst):
        os.makedirs(dst)
    for idx, f in enumerate(src_files):
        print(idx, f)
        img = np.load(f)
        # if np.sum(img==1):
        #     print(3)
        #     import matplotlib.pyplot as plt
        #     plt.imshow(img)
        #     plt.show()

        # TODO:
        # if img.dtype == bool:
        #     img = np.uint8(img)
        # else:
        #     img = np.uint8(255*((img-np.min(img))/(np.max(img)-np.min(img))))
        new_f = os.path.join(os.path.split(f)[0].replace(src, dst), os.path.split(f)[1].replace(src_format, dst_format))
        if not os.path.isdir(os.path.split(new_f)[0]):
            os.makedirs(os.path.split(new_f)[0])
        cv2.imwrite(new_f, img)


if __name__ == '__main__':
    # src = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-Preprocessing\Image'
    # dst = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-Preprocessing-png\Image'
    # src_format = 'npy'
    # dst_format = 'png'
    # convert_npy_to_png(src, dst, src_format, dst_format)

    src = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI'
    dst = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-all-slices'
    lidc_preprocess(path=src, save_path=dst)
    pass