
import os
import functools
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils.utils import mask_preprocess, raw_preprocess
import logging
from Liwei.LUNA16_test.disk import getCache
from Liwei.LUNA16_test import dataset_seg, util
raw_cache = getCache('part2segment')

logging.basicConfig(level=logging.INFO)
from data import data_utils

# TODO: unifyinterface
def get_data_by_pid_asus(data_path, pid):
    raw_and_mask = data_utils.get_files(os.path.join(data_path, pid), recursive=False, get_dirs=True)
    for _dir in raw_and_mask:
        file_dir = os.path.split(_dir)[1]
        if 'raw' in file_dir:
            raw_dir = _dir
        elif 'mask' in file_dir:
            mask_dir = _dir

    vol_raw_path = data_utils.get_files(raw_dir, 'mhd', recursive=False)[0]
    vol, origin, spacing, direction = data_utils.load_itk(vol_raw_path)
    raw_vol = vol.copy()
    raw_vol = raw_preprocess(raw_vol, output_dtype=np.int32, norm=False)
    vol = np.clip(vol, -1000, 1000)
    vol = raw_preprocess(vol, output_dtype=np.uint8)

    vol_mask_path = data_utils.get_files(mask_dir, 'mhd', recursive=False)[0]
    mask_vol, _, _, _ = data_utils.load_itk(vol_mask_path)
    mask_vol = mask_preprocess(mask_vol)        
    return raw_vol, vol, mask_vol, origin, spacing, direction    


def asus_nodule_volume_generator(data_path, case_pids=None, case_indices=None):
    # nodule_type = os.path.split(data_path)[1]
    case_list = data_utils.get_files(data_path, recursive=False, get_dirs=True)
    if case_pids is not None:
        sub_case_list = []
        for case_dir in case_list:
            if os.path.split(case_dir)[1] in case_pids:
                sub_case_list.append(case_dir)
        case_list = sub_case_list
    else:
        if case_indices is not None:
            case_list = np.take(case_list, case_indices)

    print(f'Generating {len(case_list)} cases...')
    for case_dir in case_list:
        raw_and_mask = data_utils.get_files(case_dir, recursive=False, get_dirs=True)
        assert len(raw_and_mask) == 2
           
        pid = os.path.split(case_dir)[1]
        scan_idx = int(pid[2:])
        raw_vol, vol, mask_vol, origin, spacing, direction = get_data_by_pid_asus(data_path, pid)

        # TODO: what is scan_idx
        infos = {'pid': pid, 'scan_idx': scan_idx, 'subset': None, 'origin': origin, 'spacing': spacing, 'direction': direction}
        yield raw_vol, vol, mask_vol, infos


class ASUSNoduleVolumeGenerator():
    def __init__(self, data_path=None, case_pids=None, case_indices=None):
        self.data_path = data_path
        self.case_pids = case_pids
        self.case_indices = case_indices
        self.total_num_slice = self.get_num_slice(self.data_path, self.case_pids)

    def build_volume_generator(self):
        # nodule_type = os.path.split(data_path)[1]
        case_list = data_utils.get_files(self.data_path, recursive=False, get_dirs=True)
        if self.case_pids is not None:
            sub_case_list = []
            for case_dir in case_list:
                if os.path.split(case_dir)[1] in self.case_pids:
                    sub_case_list.append(case_dir)
            case_list = sub_case_list
        else:
            if self.case_indices is not None:
                case_list = np.take(case_list, self.case_indices)

        print(f'Generating {len(case_list)} cases...')
        for case_dir in case_list:
            raw_and_mask = data_utils.get_files(case_dir, recursive=False, get_dirs=True)
            assert len(raw_and_mask) == 2
            
            pid = os.path.split(case_dir)[1]
            scan_idx = int(pid[2:])
            raw_vol, vol, mask_vol, origin, spacing, direction = self.get_data_by_pid_asus(self.data_path, pid)

            infos = {'pid': pid, 'scan_idx': scan_idx, 'subset': None, 'origin': origin, 'spacing': spacing, 'direction': direction}
            yield raw_vol, vol, mask_vol, infos

    @classmethod
    def get_data_by_pid_asus(cls, data_path, pid):
        raw_and_mask = data_utils.get_files(os.path.join(data_path, pid), recursive=False, get_dirs=True)
        for _dir in raw_and_mask:
            if 'raw' in _dir:
                raw_dir = _dir
            elif 'mask' in _dir:
                mask_dir = _dir

        vol_raw_path = data_utils.get_files(raw_dir, 'mhd', recursive=False)[0]
        vol, origin, spacing, direction = data_utils.load_itk(vol_raw_path)
        raw_vol = vol.copy()
        raw_vol = raw_preprocess(raw_vol, output_dtype=np.int32, norm=False)
        vol = np.clip(vol, -1000, 1000)
        vol = raw_preprocess(vol, output_dtype=np.uint8)

        vol_mask_path = data_utils.get_files(mask_dir, 'mhd', recursive=False)[0]
        mask_vol, _, _, _ = data_utils.load_itk(vol_mask_path)
        mask_vol = mask_preprocess(mask_vol)        
        return raw_vol, vol, mask_vol, origin, spacing, direction    

    def get_num_slice(self, data_path, case_pids):
        total_num_slice = {}
        for pid in case_pids:
            all_gen = self.get_data_by_pid_asus(data_path, pid)
            raw_vol = all_gen[0]
            total_num_slice[pid] = raw_vol.shape[0]
        return total_num_slice


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
    center = np.array(list(center))
    origin = np.array(list(origin))
    spacing = np.array(list(spacing))
    # convert to nodule spacing from world coordinates

    # Defining the voxel range in which the nodule falls
    world_range = 5 # 5 mm
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+world_range)
    v_xmin = np.max([0, int(v_center[0]-v_diam)-world_range])
    v_xmax = np.min([width-1, int(v_center[0]+v_diam)+world_range])
    v_ymin = np.max([0, int(v_center[1]-v_diam)-world_range]) 
    v_ymax = np.min([height-1, int(v_center[1]+v_diam)+world_range])
    
    v_xrange = range(v_xmin, v_xmax+1)
    v_yrange = range(v_ymin, v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for p_x in x_data:
        for p_y in y_data:
            if np.linalg.norm(center-np.array([p_x, p_y, z])) <= diam:
                mask[int((p_y-origin[1])/spacing[1]), int((p_x-origin[0])/spacing[0])] = 1.0
    return mask


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct_luna16_round_mask(series_uid)


class Ct_luna16_round_mask(dataset_seg.Ct):
    def __init__(self, series_uid):
        super().__init__(series_uid)
        self.height, self.width = self.hu_a.shape[1:]

        # df_node = pd.read_csv(luna_path+"annotations.csv")
        # df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
        # self.df_node = df_node.dropna()

    def buildAnnotationMask(self, positiveInfo_list, threshold_hu=-700):
        return self.luna16_round_mask_preprocessing(positiveInfo_list)

    def luna16_round_mask_preprocessing(self, positiveInfo_list):
        mask_vol = np.zeros_like(self.hu_a)

        for candidateInfo_tup in positiveInfo_list:
            center_irc = util.xyz2irc(
                candidateInfo_tup.center_xyz,
                self.origin_xyz,
                self.vxSize_xyz,
                self.direction_a,
            )
            for i, i_z in enumerate(np.arange(int(center_irc.index)-1,
                             int(center_irc.index)+2).clip(0, self.hu_a.shape[0]-1)): # clip prevents going out of bounds in Z

                mask = make_mask(
                    candidateInfo_tup.center_xyz, candidateInfo_tup.diameter_mm, 
                    i_z*self.vxSize_xyz[2]+self.origin_xyz[2], self.width, self.height, self.vxSize_xyz, self.origin_xyz)
                                 
            # for mask_idx in range(mask_vol.shape[0]):
            #     z_in_world = irc2xyz(mask_idx, self.origin_xyz, self.vxSize_xyz, self.direction_a)
            #     mask = make_mask(
            #         candidateInfo_tup.center_xyz, candidateInfo_tup.diameter_mm, 
            #         z_in_world, self.width, self.height, self.vxSize_xyz, self.origin_xyz)

                mask_vol[i_z] = mask
        return mask_vol


class luna16_volume_generator():
    def __init__(self, data_path=None, subset_indices=None, case_indices=None):
        self.data_path = data_path
        self.subset_indices = subset_indices
        self.case_indices = case_indices
        self.total_case_list = self.get_case_list(data_path, subset_indices, case_indices)
        self.pid_list = [os.path.split(path)[1].split('.')[0] for path in self.total_case_list]
        # self.pid_list = self.get_pid_list(data_path, subset_indices, case_indices)

    @classmethod
    def Build_DLP_luna16_volume_generator(cls, data_path=None, subset_indices=None, case_indices=None, only_nodule_slices=None):
        mask_generating_op = dataset_seg.getCt
        return cls.Build_luna16_volume_generator(mask_generating_op, data_path, subset_indices, case_indices, only_nodule_slices)

    @classmethod
    def Build_Round_luna16_volume_generator(cls, data_path=None, subset_indices=None, case_indices=None, only_nodule_slices=None):
        mask_generating_op = getCt
        return cls.Build_luna16_volume_generator(mask_generating_op, data_path, subset_indices, case_indices, only_nodule_slices)

    @classmethod   
    def Build_luna16_volume_generator(cls, mask_generating_op, data_path=None, subset_indices=None, case_indices=None, only_nodule_slices=None):
        # TODO: Cancel dependency of [data_utils.get_files]
        # TODO: use self.total_case_list to calculate
        subset_list = data_utils.get_files(data_path, 'subset', recursive=False, get_dirs=True)
        if subset_indices:
            subset_list = np.take(subset_list, subset_indices)
        
        for subset_dir in subset_list:
            case_list = data_utils.get_files(subset_dir, 'mhd', recursive=False)
            if case_indices:
                case_list = np.take(case_list, case_indices)
            for case_dir in case_list:
                subset = os.path.split(subset_dir)[-1]
                series_uid = os.path.split(case_dir)[1][:-4]
                
                raw_vol, vol, mask_vol, infos = luna16_volume_generator.get_data_by_pid(series_uid)
                infos['subset'] = subset
                yield raw_vol, vol, mask_vol, infos

    @staticmethod
    def get_case_list(data_path, subset_indices, case_indices):
        total_case_list = []
        subset_list = data_utils.get_files(data_path, 'subset', recursive=False, get_dirs=True)
        if subset_indices:
            subset_list = np.take(subset_list, subset_indices)
        
        for subset_dir in subset_list:
            case_list = data_utils.get_files(subset_dir, 'mhd', recursive=False, return_fullpath=True)
            if case_indices:
                case_list = np.take(case_list, case_indices)
            total_case_list.extend(case_list)
        return total_case_list

    @staticmethod
    # @raw_cache.memoize(typed=True)
    def get_data_by_pid(pid, mask_generating_op=dataset_seg.getCt):
        ct = mask_generating_op(pid)
        raw_vol = ct.hu_a
        mask_vol = ct.positive_mask
        
        vol = np.clip(raw_vol, -1000, 1000)
        vol = raw_preprocess(vol, output_dtype=np.uint8)
        mask_vol = mask_preprocess(mask_vol)
        infos = {'dataset': 'LUNA16', 'pid': pid, 'scan_idx': 0, 
                 'origin': ct.origin_xyz, 'spacing': ct.vxSize_xyz, 'direction': ct.direction_a}
        # TODO:
        return raw_vol, vol, mask_vol, infos


def build_pred_generator(data_generator, predictor, batch_size=1):
    for vol_idx, (vol, mask_vol, infos) in enumerate(data_generator):
        infos['vol_idx'] = vol_idx
        pid, scan_idx = infos['pid'], infos['scan_idx']
        total_outputs = []
        mask_vol = np.int32(mask_vol)
        pred_vol = np.zeros_like(mask_vol)

        for img_idx in range(0, vol.shape[0], batch_size):
            if img_idx == 0:
                print(f'\n Volume {vol_idx} Patient {pid} Scan {scan_idx} Slice {img_idx}')
            start, end = img_idx, min(vol.shape[0], img_idx+batch_size)
            img = vol[start:end]
            img_list = np.split(img, img.shape[0], axis=0)
            outputs = predictor(img_list) 
            
            for j, output in enumerate(outputs):
                total_outputs.append(output["instances"])
                
        yield pid, total_outputs, infos
