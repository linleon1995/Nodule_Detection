import os
import numpy as np
import pandas as pd
from torch import positive
from utils.volume_generator import luna16_volume_generator, asus_nodule_volume_generator, get_data_by_pid_asus
from utils.utils import get_nodule_center, irc2xyz, DataFrameTool
import cc3d

CROP_RANGE =  {'index': 48, 'row': 48, 'column': 48}

# ASUS_M_RAW_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules\malignant'

# DATA_PATH = ASUS_M_RAW_DATA_PATH
# VOLUME_GENERATOR = asus_nodule_volume_generator(DATA_PATH)
# ASUS_N_ANNOTATION_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\malignant\annotations.csv'

RAW_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules'
VOL_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess'
NEGATIVE_POSITIVE_RATIO = 10
CONNECTIVITY = 26


class LUNA16_CropRange_Builder():
    @staticmethod 
    def build_random_sample_subset(crop_range, 
                                   vol_data_path, 
                                   volume_generator, 
                                   annotation_path,
                                   negative_positive_ratio=1.0):
        file_name_key = LUNA16_CropRange_Builder.get_filename_key(crop_range)
        save_path = os.path.join(vol_data_path, f'{file_name_key}-{negative_positive_ratio}')
        positive_path = os.path.join(save_path, f'positive_IRC_{file_name_key}.csv') 
        negative_path = os.path.join(save_path, f'negative_IRC_{file_name_key}.csv') 

        # Get cropping samples
        if not os.path.isfile(positive_path) or not os.path.isfile(negative_path):
            luna16_annotations = pd.read_csv(annotation_path)
            LUNA16_CropRange_Builder.save_luna16_cropping_samples(luna16_annotations, crop_range, save_path, volume_generator)
        positive_crop_range = pd.read_csv(positive_path)
        negative_crop_range = pd.read_csv(negative_path)

        # Merge positive and negative samples and save in data_samples
        num_positive_sample = positive_crop_range.shape[0]
        num_negative_sample = int(negative_positive_ratio*num_positive_sample)
        negative_crop_range_subset = negative_crop_range.sample(n=num_negative_sample)
        data_samples = pd.concat([positive_crop_range, negative_crop_range_subset])
        data_samples.to_csv(os.path.join(save_path, f'data_samples.csv'))

        # Make directory
        positive_raw_path = os.path.join(save_path, 'positive', 'Image')
        negative_raw_path = os.path.join(save_path, 'negative', 'Image')
        positive_target_path = os.path.join(save_path, 'positive', 'Mask')
        negative_target_path = os.path.join(save_path, 'negative', 'Mask')
        for save_dir in [positive_raw_path, negative_raw_path, positive_target_path, negative_target_path]:
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

        total_raw_path = np.array([])
        for index, data_info in data_samples.iterrows():
            short_pid = data_info['seriesuid'].split('.')[-1]
            if short_pid[1] == 'B':
                nodule_type = 'benign'
            elif short_pid[1] == 'm':
                nodule_type = 'malignant'

            _, input_volume, target_volume, origin, spacing, direction = get_data_by_pid_asus(short_pid, nodule_type)
            
            crop_center = {'index': data_info['center_i'], 'row': data_info['center_r'], 'column': data_info['center_c']}
            raw_chunk = LUNA16_CropRange_Builder.crop_volume(input_volume, crop_range, crop_center)
            target_chunk = LUNA16_CropRange_Builder.crop_volume(target_volume, crop_range, crop_center)
            print(f'Saving ASUS nodule volume {index:04d} with shape {raw_chunk[...,0].shape}')

            if data_info['category'] == 'positive':
                raw_path, target_path = positive_raw_path, positive_target_path
            elif data_info['category'] == 'negative':
                raw_path, target_path = negative_raw_path, negative_target_path
                
            file_name = f'asus-{index:04d}-{short_pid}'
            np.save(os.path.join(raw_path, f'{file_name}.npy'), raw_chunk[...,0])
            np.save(os.path.join(target_path, f'{file_name}.npy'), target_chunk)
            total_raw_path = np.append(total_raw_path, os.path.join(data_info['category'], 'Image', f'{file_name}.npy'))

        data_samples['path'] = total_raw_path
        data_samples.to_csv(os.path.join(save_path, f'data_samples.csv'))

    @staticmethod
    def save_luna16_cropping_samples(luna16_annotations,
                                     crop_range, 
                                     save_path, 
                                     volume_generator):
        # e.g., LUNA16_CropRange_Builder.save_luna16_cropping_samples({'index': 64, 'row': 64, 'column': 64}, 'the path
        # where dataset save')
        total_positive, total_negative = None, None
        
        for vol_idx, (_, raw_volume, target_volume, volume_info) in enumerate(volume_generator):
            # if vol_idx >= 3: break
            print(vol_idx+1, volume_info['pid'])

            nodule_annotation = luna16_annotations.loc[luna16_annotations['seriesuid'].isin([volume_info['pid']])]
            nodule_center_xyz = nodule_annotation[['coordX', 'coordY', 'coordZ']].to_numpy()
            positive, negative = LUNA16_CropRange_Builder.get_luna16_cropping_sample(
                target_volume, crop_range, nodule_center_xyz, volume_info['origin'], volume_info['spacing'], volume_info['direction'])
            
            def add_data_sample(total_sample, sample_df, volume_info, category_key):
                num_sample = sample_df.shape[0]
                subset_array = np.array(num_sample*[volume_info['subset']])[:,np.newaxis]
                pid_array = np.array(num_sample*[volume_info['pid']])[:,np.newaxis]
                category_array = np.array(num_sample*[category_key])[:,np.newaxis]
                positive_data = np.concatenate([subset_array, pid_array, sample_df, category_array], axis=1)
                positive_df = pd.DataFrame(positive_data, columns=['subset', 'seriesuid', 'center_i', 'center_r', 'center_c', 'category'])
                total_sample = pd.concat([total_sample, positive_df]) if total_sample is not None else positive_df
                return total_sample

            if positive is not None:
                total_positive = add_data_sample(total_positive, positive, volume_info, category_key='positive')

            if negative is not None:
                total_negative = add_data_sample(total_negative, negative, volume_info, category_key='negative')

        filename_key = LUNA16_CropRange_Builder.get_filename_key(crop_range)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        total_positive.to_csv(os.path.join(save_path, f'positive_IRC_{filename_key}.csv'), index=False)
        total_negative.to_csv(os.path.join(save_path, f'negative_IRC_{filename_key}.csv'), index=False)

    @staticmethod
    def get_luna16_cropping_sample(target_volume, crop_range, nodule_center_xyz, origin_xyz, spacing_xyz, direction_xyz):
        """Get single volume cropping samples with center, crop range"""
        depth, height, width = target_volume.shape
        positive_sample, negative_samples = None, None

        # nodule centers in voxel coord.
        voxel_nodule_center_list = [LUNA16_CropRange_Builder.xyz2irc(nodule_center, origin_xyz, spacing_xyz, direction_xyz) for nodule_center in nodule_center_xyz]

        index_begin, row_begin, column_begin = crop_range['index']//2, crop_range['row']//2, crop_range['column']//2
        index_end, row_end, column_end =  depth-index_begin, height-row_begin, width-column_begin
        
        # Get positive samples
        crop_range_descend = np.sort(np.array(list(crop_range.values())))[::-1]
        max_crop_distance = np.linalg.norm(crop_range_descend[:2])
        max_crop_distance *= 2
        modify_center = lambda center, begin, end: np.clip(center, begin, end)
        # TODO: crop_range = [128,128,128] in some case index_end will be smaller than index_start due to short depth

        if len(voxel_nodule_center_list) > 0:
            # Because we cannot promise the nodule is smaller than crop range and also we don't need that much negative samples
            for nodule_center in voxel_nodule_center_list:
                nodule_center = np.array([modify_center(nodule_center[0], index_begin, index_end),
                                          modify_center(nodule_center[1], row_begin, row_end),
                                          modify_center(nodule_center[2], column_begin, column_end)])
                positive_sample = np.concatenate([positive_sample, nodule_center[np.newaxis]], axis=0) if positive_sample is not None else nodule_center[np.newaxis]

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
                        negative_samples = np.concatenate([negative_samples, candidate_center[np.newaxis]], axis=0) if negative_samples is not None else candidate_center[np.newaxis]

        return positive_sample, negative_samples

    @staticmethod         
    def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
        origin_a = np.array(origin_xyz)
        vxSize_a = np.array(vxSize_xyz)
        coord_a = np.array(coord_xyz)
        cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
        cri_a = np.round(cri_a)
        return np.array((int(cri_a[2]), int(cri_a[1]), int(cri_a[0])))

    @staticmethod         
    def get_filename_key(crop_range):
        index, row, col = crop_range['index'], crop_range['row'], crop_range['column']
        return f'{index}x{row}x{col}'

    @staticmethod
    def crop_volume(volume, crop_range, crop_center):
        def get_interval(crop_range_dim, center, size_dim):
            begin = center - crop_range_dim//2
            end = center + crop_range_dim//2
            if begin < 0:
                begin, end = 0, end-begin
            elif end > size_dim:
                modify_distance = end - size_dim + 1
                begin, end = begin-modify_distance, size_dim-1
            # print(crop_range_dim, center, size_dim, begin, end)
            assert end-begin == crop_range_dim, f'Actual cropping range {end-begin} not fit the required cropping range {crop_range_dim}'
            return (begin, end)

        index_interval = get_interval(crop_range['index'], crop_center['index'], volume.shape[0])
        row_interval = get_interval(crop_range['row'], crop_center['row'], volume.shape[1])
        column_interval = get_interval(crop_range['column'], crop_center['column'], volume.shape[2])

        return volume[index_interval[0]:index_interval[1], 
                      row_interval[0]:row_interval[1], 
                      column_interval[0]:column_interval[1]]
    

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


def save_asus_center_info(volume_generator, connectivity, save_path):
    center_df = DataFrameTool(['seriesuid', 'coordX', 'coordY', 'coordZ'])
    for vol_idx, (_, raw_volume, target_volume, volume_info) in enumerate(volume_generator):
        # if vol_idx > 2: break
        print(f'Saving Annotation of Volume {vol_idx}')
        origin_xyz, vxSize_xyz, direction = volume_info['origin'], volume_info['spacing'], volume_info['direction']
        total_nodule_center = get_nodule_center_from_volume(target_volume, connectivity, origin_xyz, vxSize_xyz, direction)
        for nodule_center in total_nodule_center:
            center_df.write_row([volume_info['pid']] + list(nodule_center))

    save_dir = os.path.split(save_path)[0]
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    center_df.save_data_frame(save_path)


def main():
    for nodule_type in ['benign', 'malignant']:
        vol_data_path = os.path.join(VOL_DATA_PATH, nodule_type, 'crop')
        DATA_PATH = os.path.join(RAW_DATA_PATH, nodule_type)
        VOLUME_GENERATOR = asus_nodule_volume_generator(DATA_PATH)
        ANNOTATION_PATH = os.path.join(DATA_PATH, 'annotations.csv')

        if not os.path.isfile(ANNOTATION_PATH):
            save_asus_center_info(volume_generator=VOLUME_GENERATOR, connectivity=CONNECTIVITY, save_path=ANNOTATION_PATH)
        LUNA16_CropRange_Builder.build_random_sample_subset(crop_range=CROP_RANGE, 
                                                            vol_data_path=vol_data_path, 
                                                            volume_generator=VOLUME_GENERATOR, 
                                                            annotation_path=ANNOTATION_PATH, 
                                                            negative_positive_ratio=NEGATIVE_POSITIVE_RATIO)
  

if __name__ == '__main__':
    img_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\malignant\crop\48x48x48-10\positive\Image\asus-0001-1m0002.npy'
    mask_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\malignant\crop\48x48x48-10\positive\Mask\asus-0001-1m0002.npy'
    img_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\malignant\crop\48x48x48-10\negative\Image\asus-0004-1m0001.npy'
    mask_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\malignant\crop\48x48x48-10\negative\Mask\asus-0004-1m0001.npy'
    vol = np.load(img_path)
    mask_vol = np.load(mask_path)
    import matplotlib.pyplot as plt
    for img, m in zip(vol, mask_vol):
        # if np.sum(m) > 0:
        plt.imshow(img, 'gray')
        plt.imshow(m, alpha=0.2)
        plt.show()
    # main()