

import os
import numpy as np
from data.data_utils import get_shift_index
from data.volume_generator import luna16_volume_generator, asus_nodule_volume_generator


def volume_to_stacked_image(save_path, volume_generator, slice_shift):
    for vol_idx, (_, vol, mask_vol, infos) in enumerate(volume_generator):
        input_dir = os.path.join(save_path, 'shift', str(slice_shift), 'input', infos['pid'])
        target_dir = os.path.join(save_path, 'shift', str(slice_shift), 'target', infos['pid'])
        for path in [input_dir, target_dir]:
            if not os.path.isdir(path):
                os.makedirs(path)

        for slice_idx in range(vol.shape[0]):
            if slice_idx%50 == 0:
                print(f'Creating data: Volume {vol_idx} Slice {slice_idx} in {save_path}')
            slice_indcies = get_shift_index(cur_index=slice_idx, index_shift=slice_shift, boundary=[0, vol.shape[0]-1])
            input_data = np.uint8(vol[slice_indcies])
            target = np.uint8(mask_vol[slice_idx])
            # print(slice_indcies, slice_idx, np.max(input_data), np.min(input_data), np.max(target))
            np.save(os.path.join(input_dir, f'Stack{slice_shift}_{slice_idx:04d}.npy'), input_data[...,0])
            np.save(os.path.join(target_dir, f'Stack{slice_shift}_{slice_idx:04d}.npy'), target)


if __name__ == '__main__':
    slice_shif = 3
    dataset_names = ['ASUS-Benign', 'ASUS-Malignant']
    data_root = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodule'
    for dataset_name in dataset_names:
        data_path = os.path.join(data_root, dataset_name, 'merge')
        save_path = os.path.join(data_root, dataset_name)
        volume_generator = asus_nodule_volume_generator(data_path=data_path)
        volume_to_stacked_image(save_path, volume_generator, slice_shif)