
import os
import functools
import numpy as np
import matplotlib.pyplot as plt
from statistics import median_high
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import cv2_imshow, calculate_malignancy, segment_lung, mask_preprocess, raw_preprocess
# import data_preprocess
# from convert_to_coco_structure import lidc_to_datacatlog_valid
import logging
import pandas as pd
from LUNA16_test import dataset_seg, util
logging.basicConfig(level=logging.INFO)

from modules.data import dataset_utils


def lidc_volume_generator(data_path, case_indices, only_nodule_slices=False):
    case_list = dataset_utils.get_files(data_path, recursive=False, get_dirs=True)
    case_list = np.array(case_list)[case_indices]
    for case_dir in case_list:
        pid = os.path.split(case_dir)[1]
        scan_list = dataset_utils.get_files(os.path.join(case_dir, rf'Image\lung\vol\npy'), 'npy')
        for scan_idx, scan_path in enumerate(scan_list):
            vol = np.load(scan_path)
            mask_vol = np.load(os.path.join(case_dir, rf'Mask\vol\npy', os.path.split(scan_path)[1]))
            mask_vol = np.where(mask_vol>=1, 1, 0)
            if only_nodule_slices:
                nodule_slice_indices = np.where(np.sum(mask_vol, axis=(0,1)))[0]
                vol = vol[...,nodule_slice_indices]
                mask_vol = mask_vol[...,nodule_slice_indices]
            infos = {'pid': pid, 'scan_idx': scan_idx}
            yield vol, mask_vol, infos


def asus_nodule_volume_generator(data_path, subset_indices=None, case_indices=None, only_nodule_slices=False):
    case_list = dataset_utils.get_files(data_path, recursive=False, get_dirs=True)
    if case_indices:
        case_list = np.take(case_list, case_indices)
    print(f'Evaluating {len(case_list)} cases...')
    for case_dir in case_list:
        raw_and_mask = dataset_utils.get_files(case_dir, recursive=False, get_dirs=True)
        assert len(raw_and_mask) == 2
        for _dir in raw_and_mask:
            if 'raw' in _dir:
                vol_path = dataset_utils.get_files(_dir, 'mhd', recursive=False)[0]
                vol, _, _ = dataset_utils.load_itk(vol_path)
                vol = np.clip(vol, -1000, 1000)
                vol = raw_preprocess(vol, output_dtype=np.uint8)
            if 'mask' in _dir:
                vol_mask_path = dataset_utils.get_files(_dir, 'mhd', recursive=False)[0]
                mask_vol, _, _ = dataset_utils.load_itk(vol_mask_path)
                mask_vol = mask_preprocess(mask_vol)            
                # mask_vol = np.swapaxes(np.swapaxes(mask_vol, 0, 1), 1, 2)
        pid = os.path.split(case_dir)[1]
        infos = {'pid': pid, 'scan_idx': 0, 'subset': None}
        yield vol, mask_vol, infos


def make_mask(center, diam, z, width, height, spacing, origin):
    '''
    Center : centers of circles px -- list of coordinates x,y,z
    diam : diameters of circles px -- diameter
    widthXheight : pixel dim of image
    spacing = mm/px conversion rate np array x,y,z
    origin = x,y,z mm np.array
    z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
    # convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+5)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5]) 
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])

    v_xrange = range(v_xmin, v_xmax+1)
    v_yrange = range(v_ymin, v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x, p_y, z])) <= diam:
                mask[int((p_y-origin[1])/spacing[1]), int((p_x-origin[0])/spacing[0])] = 1.0
    return mask


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct_luna16_round_mask(series_uid)

class Ct_luna16_round_mask(dataset_seg.Ct):
    def __init__(self, series_uid, luna_path):
        super.__init__(self, series_uid)
        self.height, self.width = self.hu_a.shape[1:]

        df_node = pd.read_csv(luna_path+"annotations.csv")
        df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
        self.df_node = df_node.dropna()

    def buildAnnotationMask(self, positiveInfo_list, threshold_hu=-700):
        self.luna16_round_mask_preprocessing(positiveInfo_list)

    def luna16_round_mask_preprocessing(self, positiveInfo_list):
        
        boundingBox_a = np.zeros_like(self.hu_a, dtype=np.bool)

        for candidateInfo_tup in positiveInfo_list:
            center_irc = util.xyz2irc(
                candidateInfo_tup.center_xyz,
                self.origin_xyz,
                self.vxSize_xyz,
                self.direction_a,
            )
            data_preprocess.make_mask(
                candidateInfo_tup.center_xyz, candidateInfo_tup.diameter_mm, center_irc['z'], self.width, self.height, self.vxSize_xyz, self.origin_xyz)


            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            index_radius = 2
            try:
                while self.hu_a[ci + index_radius, cr, cc] > threshold_hu and \
                        self.hu_a[ci - index_radius, cr, cc] > threshold_hu:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2
            try:
                while self.hu_a[ci, cr + row_radius, cc] > threshold_hu and \
                        self.hu_a[ci, cr - row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                while self.hu_a[ci, cr, cc + col_radius] > threshold_hu and \
                        self.hu_a[ci, cr, cc - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            boundingBox_a[
                 ci - index_radius: ci + index_radius + 1,
                 cr - row_radius: cr + row_radius + 1,
                 cc - col_radius: cc + col_radius + 1] = True
        self.lung_mask[self.lung_mask == 3] = 1
        self.lung_mask[self.lung_mask == 4] = 1
        self.lung_mask[self.lung_mask != 1] = 0
        mask_a = boundingBox_a & (self.hu_a > threshold_hu) & self.lung_mask.astype('bool')

        return mask_a


class luna16_volume_generator():
    def __init__(self, data_path, subset_indices=None, case_indices=None):
        self.data_path = data_path
        self.subset_indices = subset_indices
        self.case_indices = case_indices

    @classmethod
    def Build_DLP_luna16_volume_generator(cls, data_path, subset_indices=None, case_indices=None, only_nodule_slices=None):
        mask_generating_op = dataset_seg.getCt
        return cls.Build_luna16_volume_generator(data_path, mask_generating_op, subset_indices, case_indices, only_nodule_slices)

    @classmethod
    def Build_Round_luna16_volume_generator(cls, data_path, subset_indices=None, case_indices=None, only_nodule_slices=None):
        mask_generating_op = getCt
        return cls.Build_luna16_volume_generator(data_path, mask_generating_op, subset_indices, case_indices, only_nodule_slices)

    @classmethod   
    def Build_luna16_volume_generator(cls, data_path, mask_generating_op, subset_indices=None, case_indices=None, only_nodule_slices=None):
        # TODO: Cancel dependency of [dataset_utils.get_files]
        subset_list = dataset_utils.get_files(data_path, 'subset', recursive=False, get_dirs=True)
        subset_list.sort()
        if subset_indices:
            subset_list = np.take(subset_list, subset_indices)
        
        for subset_dir in subset_list:
            case_list = dataset_utils.get_files(subset_dir, 'mhd', recursive=False)
            if case_indices:
                case_list = np.take(case_list, case_indices)
            for case_dir in case_list:
                subset = os.path.split(subset_dir)[-1]
                series_uid = os.path.split(case_dir)[1][:-4]
                
                # preprocess
                ct = mask_generating_op(series_uid)
                vol = ct.hu_a
                mask_vol = ct.positive_mask
                
                vol = np.clip(vol, -1000, 1000)
                vol = raw_preprocess(vol, output_dtype=np.uint8)
                mask_vol = mask_preprocess(mask_vol)
                infos = {'dataset': 'LUNA16', 'pid': series_uid, 'scan_idx': 0, 'subset': subset}
                yield vol, mask_vol, infos


            