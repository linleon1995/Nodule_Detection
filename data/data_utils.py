import numpy as np
import os
import logging
import cv2
import json
import cc3d
import SimpleITK as sitk



def modify_array_in_itk(itkimage, new_array):
    img_preprocessed = sitk.GetImageFromArray(new_array)
    img_preprocessed.CopyInformation(itkimage)
    return img_preprocessed


def load_itk(filename):
    '''
    This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
    '''
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    ct_scan = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    direction = np.array(list(reversed(itkimage.GetDirection()))).reshape(3, 3)
    return ct_scan, origin, spacing, direction


def get_pids_from_coco(coco_paths):
    if not isinstance(coco_paths, list):
        coco_paths = [coco_paths]
    total_case_pids = []
    for coco_path in coco_paths:
        with open(coco_path, newline='') as jsonfile:
            annotation = json.load(jsonfile)
        
        case_pids = []
        for image in annotation['images']:
            case_pid = image['id'].split('_')[0]
            if case_pid not in case_pids:
                case_pids.append(case_pid)

        total_case_pids.extend(case_pids)
    return total_case_pids


def get_shift_index(cur_index, index_shift, boundary=None):
    start, end = cur_index-index_shift, cur_index+index_shift

    if boundary is not None:
        boundary_length = boundary[1] - boundary[0]
        sequence_length = 2*index_shift+1
        assert sequence_length < boundary_length
        assert boundary[1] > boundary[0]
        assert (cur_index>=0 and index_shift>=0 and boundary[0]>=0 and boundary[1]>=0)
        assert cur_index >= boundary[0]  or cur_index <= boundary[1]
        if start < boundary[0]:
            start, end = boundary[0], boundary[0]+sequence_length-1
        elif end > boundary[1]:
            start, end = boundary[1]-sequence_length+1, boundary[1]

    indices = np.arange(start, end+1)
    # print(cur_index, index_shift, boundary, indices)
    return indices


def get_files(path, keys=[], return_fullpath=True, sort=True, sorting_key=None, recursive=True, get_dirs=False, ignore_suffix=False):
    """Get all the file name under the given path with assigned keys
    Args:
        path: (str)
        keys: (list, str)
        return_fullpath: (bool)
        sort: (bool)
        sorting_key: (func)
        recursive: The flag for searching path recursively or not(bool)
    Return:
        file_list: (list)
    """
    file_list = []
    assert isinstance(keys, (list, str))
    if isinstance(keys, str): keys = [keys]
    # Rmove repeated keys
    keys = list(set(keys))

    def push_back_filelist(root, f, file_list, is_fullpath):
        f = f[:-4] if ignore_suffix else f
        if is_fullpath:
            file_list.append(os.path.join(root, f))
        else:
            file_list.append(f)

    for i, (root, dirs, files) in enumerate(os.walk(path)):
        # print(root, dirs, files)
        if not recursive:
            if i > 0: break

        if get_dirs:
            files = dirs
            
        for j, f in enumerate(files):
            if keys:
                for key in keys:
                    if key in f:
                        push_back_filelist(root, f, file_list, return_fullpath)
            else:
                push_back_filelist(root, f, file_list, return_fullpath)

    if file_list:
        if sort: file_list.sort(key=sorting_key)
    else:
        f = 'dir' if get_dirs else 'file'
        if keys: 
            logging.warning(f'No {f} exist with key {keys}.') 
        else: 
            logging.warning(f'No {f} exist.') 
    return file_list


def cv2_imshow(img, save_path=None):
    # pass
    cv2.imshow('My Image', img)
    cv2.imwrite(save_path if save_path else 'sample.png', img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
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