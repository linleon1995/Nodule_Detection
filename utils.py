import os
import stat
import cv2
import numpy as np
import site_path
import pylidc as pl
from pylidc.utils import consensus
from statistics import median_high
import cc3d
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter
from skimage import measure, morphology
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import pandas as pd

from modules.data import dataset_utils


def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    # coords_xyz = (direction_a @ (idx * vxSize_a)) + origin_a
    return coords_xyz


def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a)
    return cri_a


def get_nodule_center(nodule_volume):
    zs, ys, xs = np.where(nodule_volume)
    center_irc = np.array([np.mean(zs), np.mean(ys), np.mean(xs)])
    return center_irc


class DataFrameTool():
    def __init__(self, column_name):
        self.df = self.build_data_frame(column_name)
        self.row_index = 0

    def build_data_frame(self, column_name):
        return pd.DataFrame(columns=column_name)
    
    def write_row(self, data):
        self.df.loc[self.row_index] = data
        self.row_index += 1

    def get_data_frame(self):
        return self.df

    def save_data_frame(self, save_path, first_column_index=False):
        save_dir = os.path.split(save_path)[0]
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.df.to_csv(save_path, index=first_column_index)


class SubmissionDataFrame(DataFrameTool):
    def __init__(self):
        column_name = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability']
        super().__init__(column_name)


# TODO: Inherit from DataFrameTool
class Nodule_data_recording():
    def __init__(self):
        self.df = self.build_data_frame()
        self.nodule_idx = 0

    def build_data_frame(self):
        vol_info_attritube = ['Nodule ID', 'Nodule IoU', 'Nodule DSC', 'Slice Number', 
                                'Size', 'Relative Size','Best Slice IoU', 'Best Slice Index']
        vol_info_attritube.insert(0, 'Series uid')
        vol_info_attritube.extend(['IoU>0.1', 'IoU>0.3', 'IoU>0.5', 'IoU>0.7', 'IoU>0.9'])
        df = pd.DataFrame(columns=vol_info_attritube)
        return df

    def write_row(self, vol_nodule_infos, pid):
        for nodule_info in vol_nodule_infos:
            vol_info_value = list(nodule_info.values())
            vol_info_value.insert(0, pid)
            vol_info_value.extend([np.int32(nodule_info['Nodule IoU']>0.1), 
                                   np.int32(nodule_info['Nodule IoU']>0.3), 
                                   np.int32(nodule_info['Nodule IoU']>0.5), 
                                   np.int32(nodule_info['Nodule IoU']>0.7), 
                                   np.int32(nodule_info['Nodule IoU']>0.9)])
            self.df.loc[self.nodule_idx] = vol_info_value
            self.nodule_idx += 1
            print(nodule_info)

    def get_data_frame(self):
        return self.df


def check_image():
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\benign'
    save_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\benign\checked'
    dir_list = dataset_utils.get_files(data_path, '1B', get_dirs=True, recursive=False)
    data_index = 0
    df = pd.DataFrame(columns=['Pid', 'Slice', 'Contour num', 'Slice mask size', 'Issue'])
    for _dir in dir_list:
        raw_mask_dir = dataset_utils.get_files(_dir, get_dirs=True, recursive=False)
        for sub_dir in raw_mask_dir:
            if 'raw' in sub_dir:
                raw_path = dataset_utils.get_files(sub_dir, 'mhd')[0]
                raw_vol, _, _ = dataset_utils.load_itk(raw_path)
                raw_vol = raw_vol
            if 'mask' in sub_dir:
                mask_path = dataset_utils.get_files(sub_dir, 'mhd')[0]
                mask_vol, _, _ = dataset_utils.load_itk(mask_path)
                mask_vol = mask_vol
            
        if np.shape(raw_vol) != np.shape(mask_vol):
            print('Unmatch problem!', raw_path)
            break

        pid = os.path.split(_dir)[1]
        save_sub_dir = os.path.join(save_path, pid)
        if not os.path.isdir(save_sub_dir):
            os.makedirs(save_sub_dir)

        print(pid)
        for slice_idx, (raw_slice, mask_slice) in enumerate(zip(raw_vol, mask_vol)):
            raw_slice = raw_preprocess(raw_slice)
            mask_slice = np.uint8(mask_slice)
            if np.sum(mask_slice) > 0:
                
                mask_contours, _ = cv2.findContours(mask_slice, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # cv2.drawContours(mask_slice, contours=mask_contours, contourIdx=-1, color=(0, 0, 255))
                contour_slice = raw_slice.copy()
                cv2.drawContours(contour_slice, contours=mask_contours, contourIdx=-1, color=(255, 0, 0))

                # plt.imshow(mask_slice)
                # plt.show()

                data_list = [pid, slice_idx, len(mask_contours), np.sum(mask_slice), 'Pass']
                if len(mask_contours) > 1:
                    print('Potential Annotation problem', pid, slice_idx)
                    fig0, ax0 = plt.subplots(1,1, dpi=400)
                    ax0.imshow(mask_slice, 'gray')
                    fig0.savefig(os.path.join(save_sub_dir, f'{slice_idx}_mask.png'))
                    ax0.clear()
                    ax0.imshow(enlarge_binary_image(mask_slice), 'gray')
                    fig0.savefig(os.path.join(save_sub_dir, f'{slice_idx}_mask_en.png'))
                    fig1, _ = compare_result(raw_slice, mask_slice, mask_slice)
                    fig2, _ = compare_result_enlarge(raw_slice, mask_slice, mask_slice)
                    fig1.savefig(os.path.join(save_sub_dir, f'{slice_idx}_comp.png'))
                    fig2.savefig(os.path.join(save_sub_dir, f'{slice_idx}_comp_en.png'))
                    data_list[-1] = 'Annotation problem'
                    
                fig3, ax3 = plt.subplots(1,1)
                ax3.imshow(contour_slice, 'gray')
                ax3.set_title(f'Contour number: {len(mask_contours)}')
                # plt.show()
                fig3.savefig(os.path.join(save_sub_dir, f'{slice_idx}.png'))
                df.loc[data_index] = data_list
                data_index += 1
    df.to_csv(os.path.join(save_path, 'check.csv'))
                
            


class time_record():
    # TODO: to check the start and end be setted together

    def __init__(self):
        self.time_period = {}
        self.start_time = {}
        self.end_time = {}
        self.call_times = 1

    def set_start_time(self, name):
        self.start_time[name] = time.time()

    def set_end_time(self, name):
        self.end_time[name] = time.time()
        if name in self.time_period:
            self.time_period[name] += (self.end_time[name] - self.start_time[name])
        else:
            self.time_period[name] = self.end_time[name] - self.start_time[name]

    def show_recording_time(self):
        total_time = self.time_period['Total']
        function_time = 0
        for name, time in self.time_period.items():
            if name != 'Total':
                print(f'{name} {time:.2f} second {time/total_time*100:.2f} %')
                function_time += time

        if len(self.time_period.keys()) > 1:
            print(f'Others {total_time-function_time} second {(total_time-function_time)/total_time*100:.2f} %')
        print(f'Total {total_time:.2f} second {total_time/total_time*100:.2f} %')

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





def raw_preprocess(img, lung_segment=False, norm=True, change_channel=True, output_dtype=np.int32):
    if lung_segment:
        assert img.ndim == 2
        img = segment_lung(img)
        img[img==-0] = 0

    if norm:
        dtype = img.dtype
        img = np.float32(img) # avoid overflow
        if np.max(img)==np.min(img):
            img = np.zeros_like(img)
        else:
            img = 255*((img-np.min(img))/(np.max(img)-np.min(img)))
        img = np.array(img, dtype)
    
    if change_channel:
        img = np.tile(img[...,np.newaxis], np.append(np.ones(img.ndim, dtype=np.int32), 3))

    img = output_dtype(img)
    return img


def mask_preprocess(mask, ignore_malignancy=True, output_dtype=np.int32):
    # assert mask.ndim == 2
    if ignore_malignancy:
        mask = np.where(mask>=1, 1, 0)
    mask = output_dtype(mask)
    return mask


def enlarge_binary_image(binary_image, crop_range=30):
    if binary_image.ndim == 3:
        h, w, c = binary_image.shape
    elif binary_image.ndim == 2:
        h, w = binary_image.shape
    else:
        raise ValueError('Unknown input image shape')

    ys, xs = np.where(binary_image)
    x1, x2 = max(0, min(xs)-crop_range), min(max(xs)+crop_range, min(h,w))
    y1, y2 = max(0, min(ys)-crop_range), min(max(ys)+crop_range, min(h,w))
    bbox_size = np.max([np.abs(x1-x2), np.abs(y1-y2)])
    return cv2.resize(binary_image[y1:y1+bbox_size, x1:x1+bbox_size], (w, h), interpolation=cv2.INTER_NEAREST)


def compare_result(image, label, pred, show_mask_size=False, **imshow_params):
    if 'alpha' not in imshow_params: imshow_params['alpha'] = 0.2
    fig, ax = plt.subplots(1,2, figsize=(8,4), tight_layout=True)
    ax[0].imshow(image)
    ax[0].imshow(label, **imshow_params)
    ax[1].imshow(image)
    ax[1].imshow(pred, **imshow_params)
    ax[0].set_title(f'Label (Mask size: {np.sum(label)})' if show_mask_size else 'Label')
    ax[1].set_title(f'Prediction (Mask size: {np.sum(pred)})' if show_mask_size else 'Prediction')
    return fig, ax


def compare_result_enlarge(image, label, pred, show_mask_size=False, **imshow_params):
    crop_range = 30
    if np.sum(label) > 0:
        item = label
    else:
        if np.sum(pred) > 0:
            item = pred
        else:
            item = None
    
    fig, ax = None, None
    if item is not None:
        if image.ndim == 2:
            image = raw_preprocess(image, lung_segment=False, norm=False)
        image = np.uint8(image)
        h, w, c = image.shape
        ys, xs = np.where(item)
        x1, x2 = max(0, min(xs)-crop_range), min(max(xs)+crop_range, min(h,w))
        y1, y2 = max(0, min(ys)-crop_range), min(max(ys)+crop_range, min(h,w))
        bbox_size = np.max([np.abs(x1-x2), np.abs(y1-y2)])

        image = cv2.resize(image[y1:y1+bbox_size, x1:x1+bbox_size], (w, h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label[y1:y1+bbox_size, x1:x1+bbox_size], (w, h), interpolation=cv2.INTER_NEAREST)
        pred = cv2.resize(pred[y1:y1+bbox_size, x1:x1+bbox_size], (w, h), interpolation=cv2.INTER_NEAREST)

        fig, ax = compare_result(image, label, pred, show_mask_size, **imshow_params)
    return fig, ax


def cv2_imshow(img, save_path=None):
    # pass
    cv2.imshow('My Image', img)
    cv2.imwrite(save_path if save_path else 'sample.png', img)
    # cv2.waitKey(0)
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


def split_individual_mask(mask):
    individual_mask = cc3d.connected_components(mask, connectivity=8)
    if np.max(individual_mask) > 1:
        mask_list = []
        for mask_label in range(1, np.max(individual_mask)):
            mask_list.append(np.uint8(individual_mask==mask_label))
        return mask_list
    else:
        return [mask]


def merge_near_masks(sub_masks, distance_threshold=128):
    candidate_pool = []
    for mask in sub_masks:
        if not candidate_pool:
            candidate_pool.append(mask)
        else:
            for candidate_idx, mask_candidate in enumerate(candidate_pool):
                ys, xs = np.where(mask)
                ys2, xs2 = np.where(mask_candidate)
                mask_min_point = np.array([np.min(ys), np.min(xs)])
                candidate_min_point = np.array([np.min(ys2), np.min(xs2)])
                mask_distance =  np.linalg.norm(mask_min_point-candidate_min_point)
                # print('distance', mask_distance)
                if mask_distance < distance_threshold:
                    candidate_pool[candidate_idx] = mask_candidate + mask
                    append_flag = False
                    break
                else:
                    append_flag = True
            
            if append_flag:
                candidate_pool.append(mask)
    return candidate_pool

