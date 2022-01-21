import os
import numpy as np
import pandas as pd
from volume_generator import luna16_volume_generator, asus_nodule_volume_generator

CROP_RANGE =  {'index': 64, 'row': 64, 'column': 64}
VOLUME_GENERATOR = luna16_volume_generator.Build_DLP_luna16_volume_generator()
VOL_DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\crop'


class LUNA16_CropRange_Builder():
    @staticmethod 
    def build_random_sample_subset(crop_range, vol_data_path, volume_generator, positive_to_negative_ratio=1.0):
        file_name_key = LUNA16_CropRange_Builder.get_filename_key(crop_range)
        save_path = os.path.join(vol_data_path, file_name_key)
        positive_path = os.path.join(save_path, f'positive_IRC_{file_name_key}.csv') 
        negative_path = os.path.join(save_path, f'negative_IRC_{file_name_key}.csv') 

        if not os.path.isfile(positive_path) or not os.path.isfile(negative_path):
            LUNA16_CropRange_Builder.build_luna16_crop_dataset(crop_range, save_path, volume_generator)
        positive_crop_range = pd.read_csv(positive_path)
        negative_crop_range = pd.read_csv(negative_path)

        num_positive_sample = positive_crop_range.shape[0]
        num_negative_sample = int((1/positive_to_negative_ratio)*num_positive_sample)
        # TODO: negative select way: shuffle and select first num_negative_sample sample
        # TODO: the folder should be indepedent (move and use instantly) (consider relative path)
        data_samples = pd.concat([positive_crop_range, negative_crop_range.sample(n=num_negative_sample)])
        data_samples.to_csv(os.path.join(save_path, f'data_samples.csv'))

        positive_raw_path = os.path.join(save_path, 'positive', 'Image')
        negative_raw_path = os.path.join(save_path, 'negative', 'Image')
        positive_target_path = os.path.join(save_path, 'positive', 'Mask')
        negative_target_path = os.path.join(save_path, 'negative', 'Mask')
        for save_dir in [positive_raw_path, negative_raw_path, positive_target_path, negative_target_path]:
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

        total_raw_path = np.array([])
        for index, data_info in data_samples.iterrows():
            print(f'Saving LUNA16 nodule volume {index:04d}')
            pid = data_info['seriesuid']
            raw_volume, target_volume, volume_info = luna16_volume_generator.get_data_by_pid(pid)
            
            crop_center = {'index': data_info['center_i'], 'row': data_info['center_r'], 'column': data_info['center_c']}
            raw_chunk = LUNA16_CropRange_Builder.crop_volume(raw_volume, crop_range, crop_center)
            target_chunk = LUNA16_CropRange_Builder.crop_volume(target_volume, crop_range, crop_center)

            if data_info['category'] == 'positive':
                raw_path = positive_raw_path
                target_path = positive_target_path
            elif data_info['category'] == 'negative':
                raw_path = negative_raw_path
                target_path = negative_target_path
                
            np.save(os.path.join(raw_path, f'luna16-{index:04d}-{pid}.npy'), raw_chunk[...,0])
            np.save(os.path.join(target_path, f'luna16-{index:04d}-{pid}.npy'), target_chunk)
            total_raw_path = np.append(total_raw_path, os.path.join(raw_path, f'luna16-{index:04d}-{pid}.npy'))

        data_samples['path'] = total_raw_path
        data_samples.to_csv(os.path.join(save_path, f'data_samples.csv'))

    @staticmethod
    def build_luna16_crop_dataset(crop_range, save_path, volume_generator=luna16_volume_generator.Build_DLP_luna16_volume_generator()):
        # e.g., LUNA16_CropRange_Builder.build_luna16_crop_dataset({'index': 64, 'row': 64, 'column': 64}, 'crop_dataset')
        # crop_range_array = np.array((crop_range['index'], crop_range['row'], crop_range['column']))
        total_positive, total_negative = None, None
        annotations = pd.read_csv('evaluationScript/annotations/annotations.csv')
        
        for vol_idx, (raw_volume, target_volume, volume_info) in enumerate(volume_generator):
            # if vol_idx >= 80: break
            pid = volume_info['pid']
            print(vol_idx+1, pid)
            # positive, negative = LUNA16_CropRange_Builder.get_luna16_crop_range(target_volume, crop_range)

            # ++
            nodule_annotation = annotations.loc[annotations['seriesuid'].isin([pid])]
            nodule_center_xyz = nodule_annotation[['coordX', 'coordY', 'coordZ']].to_numpy()
            positive, negative = LUNA16_CropRange_Builder.get_luna16_crop_range_center(
                raw_volume, target_volume, crop_range, nodule_center_xyz, volume_info['origin'], volume_info['spacing'], volume_info['direction'])
            # ++
            
            # TODO: merge positive and negative
            if positive is not None:
                num_positive = positive.shape[0]
                subset_array = np.array(num_positive*[volume_info['subset']])[:,np.newaxis]
                pid_array = np.array(num_positive*[pid])[:,np.newaxis]
                category_array = np.array(num_positive*['positive'])[:,np.newaxis]
                positive_data = np.concatenate([subset_array, pid_array, positive, category_array], axis=1)
                positive_df = pd.DataFrame(positive_data, columns=['subset', 'seriesuid', 'center_i', 'center_r', 'center_c', 'category'])
                total_positive = pd.concat([total_positive, positive_df]) if total_positive is not None else positive_df

            if negative is not None:
                num_negative = negative.shape[0]
                subset_array = np.array(num_negative*[volume_info['subset']])[:,np.newaxis]
                pid_array = np.array(num_negative*[pid])[:,np.newaxis]
                category_array = np.array(num_negative*['negative'])[:,np.newaxis]
                negative_data = np.concatenate([subset_array, pid_array, negative, category_array], axis=1)
                negative_df = pd.DataFrame(negative_data, columns=['subset', 'seriesuid', 'center_i', 'center_r', 'center_c', 'category'])
                total_negative = pd.concat([total_negative, negative_df]) if total_negative is not None else negative_df

        filename_key = LUNA16_CropRange_Builder.get_filename_key(crop_range)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        total_positive.to_csv(os.path.join(save_path, f'positive_IRC_{filename_key}.csv'), index=False)
        total_negative.to_csv(os.path.join(save_path, f'negative_IRC_{filename_key}.csv'), index=False)

    @staticmethod
    def get_luna16_crop_range(target_volume, crop_range):
        depth, height, width = target_volume.shape
        positive_sample, negative_samples = None, None

        for candidate_center_index in range(crop_range['index']//2, depth-crop_range['index']//2, crop_range['index']):
            for candidate_center_row in range(crop_range['row']//2, height-crop_range['row']//2, crop_range['row']):
                for candidate_center_column in range(crop_range['column']//2, width-crop_range['column']//2, crop_range['column']):
                    candidate_center = np.array([candidate_center_index, candidate_center_row, candidate_center_column])
                    candidate_center_in_dict = {'index': candidate_center_index, 'row': candidate_center_row, 'column': candidate_center_column}
                    crop_target_volume = LUNA16_CropRange_Builder.crop_volume(target_volume, crop_range, candidate_center_in_dict)
                    if np.sum(crop_target_volume) > 0:
                        positive_sample = np.concatenate([positive_sample, candidate_center[np.newaxis]], axis=0) if positive_sample is not None else candidate_center[np.newaxis]
                    else:
                        negative_samples = np.concatenate([negative_samples, candidate_center[np.newaxis]], axis=0) if negative_samples is not None else candidate_center[np.newaxis]
  
        return positive_sample, negative_samples

    @staticmethod
    def get_luna16_crop_range_center(raw_volume, target_volume, crop_range, nodule_center_xyz, origin_xyz, spacing_xyz, direction_xyz):
        
        depth, height, width = target_volume.shape
        positive_sample, negative_samples = None, None

        # nodule centers in voxel coord.
        voxel_nodule_center_list = [LUNA16_CropRange_Builder.xyz2irc(nodule_center, origin_xyz, spacing_xyz, direction_xyz) for nodule_center in nodule_center_xyz]

        # TODO: nodule range (record if bigger than crop_range)

        
        # index_begin, row_begin, column_begin = int(crop_range['index']*1.5), int(crop_range['row']*1.5), int(crop_range['column']*1.5)
        # index_end, row_end, column_end =  depth-index_begin, height-row_begin, width-column_begin

        index_begin, row_begin, column_begin = crop_range['index']//2, crop_range['row']//2, crop_range['column']//2
        index_end, row_end, column_end =  depth-index_begin, height-row_begin, width-column_begin
        
        # Get positive samples
        max_gap_distance = 0
        modify_center = lambda center, begin, end: np.clip(center, begin, end)
        if len(voxel_nodule_center_list) > 0:
            # Because we cannot promise the nodule is smaller than crop range and also we don't need that much negative samples
            for nodule_center in voxel_nodule_center_list:
                nodule_center = np.array([modify_center(nodule_center[0], index_begin, index_end),
                                          modify_center(nodule_center[1], row_begin, row_end),
                                          modify_center(nodule_center[2], column_begin, column_end)])
                distance = 2 * np.linalg.norm(nodule_center)
                if distance > max_gap_distance:
                    max_gap_distance = distance
                positive_sample = np.concatenate([positive_sample, nodule_center[np.newaxis]], axis=0) if positive_sample is not None else nodule_center[np.newaxis]
            max_gap_distance *= 2 # bigger gap between positive and negative

        # Get negative samples
        for candidate_center_index in range(index_begin, index_end, crop_range['index']):
            for candidate_center_row in range(row_begin, row_end, crop_range['row']):
                for candidate_center_column in range(column_begin, column_end, crop_range['column']):
                    candidate_center = np.array([candidate_center_index, candidate_center_row, candidate_center_column])
                    overlap_with_positive = False
                    for nodule_center in voxel_nodule_center_list:
                        if np.linalg.norm(nodule_center-candidate_center) < max_gap_distance:
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
        get_interval = lambda range, center: (center-range//2, center+range//2)

        index_interval = get_interval(crop_range['index'], crop_center['index'])
        row_interval = get_interval(crop_range['row'], crop_center['row'])
        column_interval = get_interval(crop_range['column'], crop_center['column'])

        return volume[index_interval[0]:index_interval[1], 
                      row_interval[0]:row_interval[1], 
                      column_interval[0]:column_interval[1]]
    

def main():
    LUNA16_CropRange_Builder.build_random_sample_subset(CROP_RANGE, VOL_DATA_PATH, VOLUME_GENERATOR)
    
    # raw_path = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\crop\backup\center1\luna16-0038-1.3.6.1.4.1.14519.5.2.1.6279.6001.227962600322799211676960828223.npy'
    # v = np.load(raw_path)
    # print(v.shape)
    # m = np.load(mask_path)
    # import matplotlib.pyplot as plt
    # for i in range(64):
    #     if np.sum(m[i])>0:
    #         plt.imshow(v[i], 'gray')
    #         plt.imshow(m[i], alpha=0.2)
    #         plt.title(f'{i}')
    #         plt.show()

    # from modules.data import dataset_utils
    # files = dataset_utils.get_files(rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\crop\backup\center1', 'npy', recursive=False)
    # for f in files:
    #     v = np.load(f)
    #     print(v.shape)

if __name__ == '__main__':
    main()