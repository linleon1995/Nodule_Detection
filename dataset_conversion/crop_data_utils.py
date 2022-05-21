from ast import ExtSlice
from genericpath import exists
import os
import numpy as np
import pandas as pd
from torch import positive
from data.volume_generator import  asus_nodule_volume_generator, get_data_by_pid_asus
from utils.utils import get_nodule_center, DataFrameTool
from dataset_conversion.coord_transform import xyz2irc, irc2xyz
import cc3d



def build_crop_data(data_path, 
                    crop_range, 
                    vol_data_path, 
                    volume_generator_builder, 
                    annotation_path,
                    negative_positive_ratio=1.0,
                    center_shift=True,
                    shift_step=2,
                    context_threshold=0.5,
                    df_random_seed=1):
    """Crop and save the volume by using pre-saved croppping information"""
    filename_key = get_filename_key(crop_range, negative_positive_ratio, center_shift, shift_step)
    save_path = os.path.join(vol_data_path, f'{filename_key}')
    pos_csv_path = os.path.join(save_path, f'positive_IRC_{filename_key}.csv')
    neg_csv_pat = os.path.join(save_path, f'negative_IRC_{filename_key}.csv')
    # volume_generator = volume_generator_builder
    volume_generator = volume_generator_builder.build()

    # Get cropping samples
    annotations = pd.read_csv(annotation_path)
    if not os.path.isfile(pos_csv_path) or not os.path.isfile(neg_csv_pat):
        save_crop_info(
            annotations, crop_range, save_path, volume_generator, 
            negative_positive_ratio, center_shift, shift_step
        )
    positive_crop_range = pd.read_csv(pos_csv_path)
    negative_crop_range = pd.read_csv(neg_csv_pat)

    # Merge positive and negative samples and save in data_samples
    num_positive_sample = positive_crop_range.shape[0]
    num_negative_sample = int(negative_positive_ratio*num_positive_sample)
    num_negative_sample = np.clip(num_negative_sample, 0, negative_crop_range.shape[0])
    negative_crop_range_subset = negative_crop_range.sample(n=num_negative_sample, random_state=df_random_seed)
    data_samples = pd.concat([positive_crop_range, negative_crop_range_subset], ignore_index=True)

    # # Add 'path' in DataFrame
    # total_raw_path, total_file_name = [], []
    # for index, data_info in data_samples.iterrows():
    #     short_pid = data_info['seriesuid'].split('.')[-1]
    #     crop_idx = data_info['crop_idx']
    #     file_name = f'{index:04d}-{short_pid}-{crop_idx}'
    #     file_path = os.path.join(data_info['category'], 'Image', f'{file_name}.npy')
    #     total_file_name.append(file_name)
    #     total_raw_path.append(file_path)
    # data_samples['path'] = total_raw_path
    # data_samples.to_csv(os.path.join(save_path, f'data_samples.csv'))

    # Make directory
    positive_raw_path = os.path.join(save_path, 'positive', 'Image')
    negative_raw_path = os.path.join(save_path, 'negative', 'Image')
    positive_target_path = os.path.join(save_path, 'positive', 'Mask')
    negative_target_path = os.path.join(save_path, 'negative', 'Mask')
    for save_dir in [positive_raw_path, negative_raw_path, positive_target_path, negative_target_path]:
        os.makedirs(save_dir, exist_ok=True)

    drop_index = []
    num_sample = data_samples.shape[0]
    total_raw_path, total_malignancy = [], []
    for sample_idx, (index, data_info) in enumerate(data_samples.iterrows(), 1):
        # if sample_idx >= 10: break
        pid = data_info['seriesuid']
        short_pid = pid.split('.')[-1]

        # TODO:
        file_name = f'{index:04d}-{short_pid}'
        file_path = os.path.join(data_info['category'], 'Image', f'{file_name}.npy')
        total_raw_path.append(file_path)
        
        if data_info['category'] == 'positive':
            raw_path, target_path = positive_raw_path, positive_target_path
        elif data_info['category'] == 'negative':
            raw_path, target_path = negative_raw_path, negative_target_path
            
        raw_file = os.path.join(raw_path, f'{file_name}.npy')
        target_file = os.path.join(target_path, f'{file_name}.npy')
        if not os.path.isfile(raw_file) or not os.path.isfile(target_file):
            print(f'{sample_idx}/{num_sample} Saving TMH nodule volume {index:04d} with shape {crop_range}')
            # _, input_volume, target_volume, origin, spacing, direction = get_data_by_pid_asus(data_path, short_pid)
            _, input_volume, target_volume, _ = volume_generator_builder.get_data_by_pid(pid)
            crop_center = {'index': data_info['center_i'], 
                           'row': data_info['center_r'], 
                           'column': data_info['center_c']}

            raw_chunk = crop_volume(input_volume, crop_range, crop_center)
            raw_chunk = raw_chunk[...,0]
            is_rich, rich_ratio = is_rich_context(raw_chunk, context_threshold, return_ratio=True)
            if data_info['category'] == 'positive':
                is_rich = True

            if is_rich:
                malignancy = is_malignancy(target_chunk)
                total_malignancy.append(malignancy)
                if not os.path.isfile(raw_file):
                    np.save(os.path.join(raw_path, f'{file_name}.npy'), raw_chunk)

                if not os.path.isfile(target_file):
                    target_chunk = crop_volume(target_volume, crop_range, crop_center)
                    np.save(os.path.join(target_path, f'{file_name}.npy'), target_chunk)
            else:
                print(f'-- Drop out index {index} context_rich {rich_ratio} category {data_info["category"]}\n')
                drop_index.append(index)

    # data_samples.drop(list(np.arange(1172)), inplace=True)
    data_samples['path'] = total_raw_path        
    data_samples['malignancy'] = total_malignancy        
    data_samples.drop(drop_index, inplace=True)
    data_samples.to_csv(os.path.join(save_path, f'data_samples.csv'))


def save_crop_info(annotations,
                   crop_range, 
                   save_path, 
                   volume_generator,
                   negative_positive_ratio, 
                   center_shift, 
                   shift_step):
    """
    Save the croppping information of assign dataset
    e.g., save_crop_info({'index': 64, 'row': 64, 'column': 64}, 'the path
    where dataset save')
    """
    total_positive, total_negative = None, None
    total_positive = []   
    total_negative = []   
    for vol_idx, (_, raw_volume, target_volume, volume_info) in enumerate(volume_generator):
        # if vol_idx >= 10: break
        print(f'Saving cropping information {vol_idx+1} {volume_info["pid"]}')

        nodule_annotation = annotations.loc[annotations['seriesuid'].isin([volume_info['pid']])]
        nodule_center_xyz = nodule_annotation[['coordX', 'coordY', 'coordZ']].to_numpy()
        volum_shape = target_volume.shape
        positive_center, negative_center = get_crop_info(
            volum_shape, crop_range, nodule_center_xyz, volume_info['origin'], volume_info['spacing'], 
            volume_info['direction'], center_shift, shift_step
        )
        
        def add_data_sample(crop_center, volume_info, category_key):
            num_sample = crop_center.shape[0]
            subset_array = np.array(num_sample*[volume_info['subset']])[:,np.newaxis]
            pid_array = np.array(num_sample*[volume_info['pid']])[:,np.newaxis]
            category_array = np.array(num_sample*[category_key])[:,np.newaxis]
            crop_indices = np.arange(crop_center.shape[0])[:,None]
            row_data = np.concatenate([subset_array, pid_array, crop_indices, crop_center, category_array], axis=1)
            row_df = pd.DataFrame(row_data)
            # row_data = pd.DataFrame(positive_data, columns=[
            #     'subset', 'seriesuid', 'crop_idx', 'center_i', 'center_r', 'center_c', 'category'])
            # total_sample = pd.concat([total_sample, row_data]) if total_sample is not None else row_data
            return row_df

        if positive_center is not None:
            row_df = add_data_sample(positive_center, volume_info, category_key='positive')
            total_positive.append(row_df)

        if negative_center is not None:
            row_df = add_data_sample(negative_center, volume_info, category_key='negative')
            total_negative.append(row_df)

    column_name = ['subset', 'seriesuid', 'crop_idx', 'center_i', 'center_r', 'center_c', 'category']
    total_positive = pd.concat(total_positive)
    total_negative = pd.concat(total_negative)
    total_positive.columns = column_name
    total_negative.columns = column_name

    filename_key = get_filename_key(crop_range, negative_positive_ratio, center_shift, shift_step)
    os.makedirs(save_path, exist_ok=True)
    
    pos_csv = os.path.join(save_path, f'positive_IRC_{filename_key}.csv')
    neg_csv = os.path.join(save_path, f'negative_IRC_{filename_key}.csv')
    total_positive.to_csv(pos_csv, index=False)
    total_negative.to_csv(neg_csv, index=False)


def get_crop_info(volum_shape, crop_range, nodule_center_xyz, origin_xyz, spacing_xyz, direction_xyz, 
                  center_shift, shift_step):
    """Get the cropping information of single volume"""
    depth, height, width = volum_shape
    positive_sample, negative_samples = None, None

    # nodule centers in voxel coord.
    # TODO: 
    voxel_nodule_center_list = [
        xyz2irc(nodule_center[::-1], origin_xyz, spacing_xyz, direction_xyz) for nodule_center in nodule_center_xyz]
    # voxel_nodule_center_list = [
    #     xyz2irc(nodule_center, origin_xyz, spacing_xyz, direction_xyz)[::-1] for nodule_center in nodule_center_xyz]

    index_begin, row_begin, column_begin = crop_range['index']//2, crop_range['row']//2, crop_range['column']//2
    index_end, row_end, column_end =  depth-index_begin, height-row_begin, width-column_begin

    # # TODO: function
    # index_range_shift, row_range_shift, col_range_shift = 0, 100 , 100
    # index_begin, index_end = index_begin+index_range_shift, index_end-index_range_shift
    # row_begin, row_end = row_begin+row_range_shift, row_end-row_range_shift
    # column_begin, column_end = column_begin+col_range_shift, column_end-col_range_shift
    
    # Get positive samples
    crop_range_descend = np.sort(np.array(list(crop_range.values())))[::-1]
    max_crop_distance = np.linalg.norm(crop_range_descend[:2])
    max_crop_distance *= 2
    modify_center = lambda center, begin, end: np.clip(center, begin, end)
    # TODO: crop_range = [128,128,128] in some case index_end will be smaller than index_start due to short depth

    if len(voxel_nodule_center_list) > 0:
        # Because we cannot promise the nodule is smaller than crop range and also we don't need that much 
        # negative samples
        for nodule_center in voxel_nodule_center_list:
            if center_shift and shift_step>0:
                for index_shift in [-shift_step, shift_step]:
                    for row_shift in [-shift_step, shift_step]:
                        for column_shift in [-shift_step, shift_step]:
                            # if center_shift ==0 and index_shift == 0 and column_shift == 0:
                            #     continue
                            nodule_center[0] = nodule_center[0] + index_shift
                            nodule_center[1] = nodule_center[1] + row_shift
                            nodule_center[2] = nodule_center[2] + column_shift

                            nodule_center = np.array([
                                modify_center(nodule_center[0], index_begin, index_end),
                                modify_center(nodule_center[1], row_begin, row_end),
                                modify_center(nodule_center[2], column_begin, column_end)])
                            positive_sample = np.concatenate(
                                    [positive_sample, nodule_center[np.newaxis]], axis=0
                                ) \
                                if positive_sample is not None else nodule_center[np.newaxis]
            else:       
                nodule_center = np.array([
                    modify_center(nodule_center[0], index_begin, index_end),
                    modify_center(nodule_center[1], row_begin, row_end),
                    modify_center(nodule_center[2], column_begin, column_end)])
                positive_sample = np.concatenate(
                        [positive_sample, nodule_center[np.newaxis]], axis=0
                    ) \
                    if positive_sample is not None else nodule_center[np.newaxis]

    # Get negative samples
    for candidate_center_index in range(index_begin, index_end, crop_range['index']):
        for candidate_center_row in range(row_begin, row_end, crop_range['row']):
            for candidate_center_column in range(column_begin, column_end, crop_range['column']):
                candidate_center = np.array([candidate_center_index, candidate_center_row, candidate_center_column])
                overlap_with_positive = False
                for nodule_center in voxel_nodule_center_list:
                    distance_btw_positive_negative = np.linalg.norm(nodule_center-candidate_center)
                    if distance_btw_positive_negative < max_crop_distance:
                        overlap_with_positive = True
                        break
                
                if not overlap_with_positive:
                    negative_samples = np.concatenate(
                            [negative_samples, candidate_center[np.newaxis]], axis=0
                        ) \
                        if negative_samples is not None else candidate_center[np.newaxis]
    
    return positive_sample, negative_samples


def get_filename_key(crop_range, negative_positive_ratio, shift, shift_step):
    index, row, col = crop_range['index'], crop_range['row'], crop_range['column']
    file_key = f'{index}x{row}x{col}-{negative_positive_ratio}'
    if shift:
        assert shift_step > 0
        file_key = f'{file_key}-shift-{shift_step}'
    return file_key


# TODO: general format
def crop_volume(volume, crop_range, crop_center):
    def get_interval(crop_range_dim, center, size_dim):
        begin = center - crop_range_dim/2
        end = center + crop_range_dim/2
        if begin < 0:
            begin, end = 0, end-begin
        elif end > size_dim:
            modify_distance = end - size_dim + 1
            begin, end = begin-modify_distance, size_dim-1
        # print(crop_range_dim, center, size_dim, begin, end)
        begin, end = int(begin), int(end)
        assert end-begin == crop_range_dim, \
            f'Actual cropping range {end-begin} not fit the required cropping range {crop_range_dim}'
        return slice(begin, end)

    index_slice = get_interval(crop_range['index'], crop_center['index'], volume.shape[0])
    row_slice = get_interval(crop_range['row'], crop_center['row'], volume.shape[1])
    column_slice = get_interval(crop_range['column'], crop_center['column'], volume.shape[2])

    return volume[index_slice][:,row_slice][:,:,column_slice]


def is_rich_context(data, threshold=0.5, return_ratio=True):
    if data.min == data.max:
        if return_ratio:
            return (False, 0.0)
        else:
            return False

    _, counts = np.unique(data, return_counts=True)
    ratio = (data.size-max(counts)) / data.size
 
    if ratio < threshold:
        is_rich = False
    else:
        is_rich = True
    
    if return_ratio:
        return (is_rich, ratio)   
    else:
        return is_rich


def is_malignancy(data):
    max_label = data.max
    if max_label == 1:
        status = 'malignancy'
    elif max_label == 1:
        status = 'benign'
    elif max_label == 0:
        status = 'null'
    else:
        raise ValueError(f'Unknown target class {max_label}')

def get_nodule_center_from_volume(volume, connectivity, origin_xyz, vxSize_xyz, direction):
    volume = cc3d.connected_components(volume, connectivity=connectivity)
    categories = np.unique(volume)[1:]
    total_nodule_center = []
    for label in categories:
        nodule = np.where(volume==label, 1, 0)
        center_irc = get_nodule_center(nodule)
        center_xyz = irc2xyz(center_irc, origin_xyz, vxSize_xyz, direction)
        total_nodule_center.append(center_xyz)
    return total_nodule_center


def save_center_info(volume_generator, connectivity, save_path):
    center_df = DataFrameTool(['seriesuid', 'coordX', 'coordY', 'coordZ'])
    total_nodule_num = 0
    for vol_idx, (_, raw_volume, target_volume, volume_info) in enumerate(volume_generator, 1):
        # if vol_idx > 2: break
        print(f'Saving Annotation of Volume {vol_idx}')
        origin_xyz, vxSize_xyz, direction = volume_info['origin'], volume_info['spacing'], volume_info['direction']
        total_nodule_center = get_nodule_center_from_volume(
            target_volume, connectivity, origin_xyz, vxSize_xyz, direction)
        for nodule_center in total_nodule_center:
            center_df.write_row([volume_info['pid']] + list(nodule_center))
            print(nodule_center)
            total_nodule_num += 1
    print(f'Total nodule number {total_nodule_num}')
    save_dir = os.path.split(save_path)[0]
    os.makedirs(save_dir, exist_ok=True)
    center_df.save_data_frame(save_path)
