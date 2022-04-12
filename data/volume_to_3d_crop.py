from distutils.log import info
import os

# TODO: consider remove zeros

import numpy as np
from pyrsistent import v
from data.volume_generator import asus_nodule_volume_generator


class CropVolume():
    def __init__(self, crop_size, crop_shift, convert_dtype=None, overlapping=1.0):
        self.crop_size = crop_size
        self.crop_shift = crop_shift
        self.convert_dtype = convert_dtype
        self.overlapping = overlapping

    def __call__(self, volume):
        crop_data = []
        if self.convert_dtype is not None:
            volume = self.convert_dtype(volume)
        crop_slices = self.get_crop_slice(volume.shape)
        
        for slice_range in crop_slices:
            crop_slice = [slice(*r) for r in slice_range]
            crop = volume[crop_slice]
            crop_data.append({'slice': slice_range, 'data':crop})
        return crop_data

    def get_crop_slice(self, volume_shape):
        slices = []
        for dim_idx, length in enumerate(volume_shape):
            slices.append(self.simple_slice(length, self.crop_shift[dim_idx], self.crop_size[dim_idx], self.overlapping))

        # TODO: for arbitrary dimensions
        slice_comb = []
        for h in slices[0]:
            for w in slices[1]:
                for c in slices[2]: 
                    slice_comb.append([h, w, c])
        return slice_comb

    @staticmethod
    def simple_slice(length, shift, crop_length, overlapping):
        slices = []
        crop_length = int(crop_length*overlapping)
        for start in range(shift, length-shift, crop_length):
            # slices.append(slice(start, start+crop_length))
            if start+crop_length > length:
                start, end = length-crop_length, length
            else:
                start, end = start, start+crop_length
            slices.append((start, end))
        return slices


def build_crop_data(volume_generator, cropping_op, save_dir):
    p_samples, n_samples = 0, 0
    for vol_idx, (raw_vol, vol, mask_vol, infos) in enumerate(volume_generator):
        pid = infos['pid']
        print(f'{save_dir}-{pid}')

        def save_vol(vol, dir_key):
            p_samples, n_samples = 0, 0
            vol = np.swapaxes(np.swapaxes(vol, 0, 1), 1, 2)
            crop_data_list = cropping_op(vol)
            save_path = os.path.join(save_dir, dir_key, pid)
            os.makedirs(save_path, exist_ok=True)
            for crop_idx, crop_data in enumerate(crop_data_list):
                if np.sum(crop_data['data'])>0:
                    p_samples += 1
                else:
                    n_samples += 1
                np.save(os.path.join(save_path, f'{pid}-{crop_idx:03d}.npy'), crop_data['data'])
            return p_samples, n_samples

        _, _ = save_vol(vol[...,0], dir_key='Image')
        p, n = save_vol(mask_vol, dir_key='Mask')
        p_samples, n_samples = p_samples + p, n_samples + n
    p_rate = 100*p_samples/(p_samples+n_samples)
    print(f'positive rate {p_rate:.02f}')


def build_tmh_crop_data(crop_range, crop_shift, convert_dtype, save_dir, overlapping):
    cropping_op = CropVolume(crop_range, crop_shift, convert_dtype, overlapping)
    crop_range_key = 'x'.join([str(s) for s in list(crop_range)])
    crop_shift_key = 'x'.join([str(s) for s in list(crop_shift)])
    overlapping_key = str(overlapping)
    dirname = '-'.join([crop_range_key, crop_shift_key, overlapping_key])

    # Malignant
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodule\ASUS-Malignant\merge'
    case_pids = [f'1m{idx:04d}' for idx in range(1, 45)]
    volume_generator = asus_nodule_volume_generator(data_path, case_pids=case_pids)
    build_crop_data(volume_generator, cropping_op, os.path.join(save_dir, 'TMH-Malignant', dirname))

    # Benign
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodule\ASUS-Benign\merge'
    case_pids = [f'1B{idx:04d}' for idx in range(1, 26)]
    volume_generator = asus_nodule_volume_generator(data_path, case_pids=case_pids)
    build_crop_data(volume_generator, cropping_op, os.path.join(save_dir, 'TMH-Benign', dirname))




if __name__ == '__main__':
    # crop_ops = CropVolume((32,64,64), (0, 100, 100))
    # np_data = np.zeros((100, 512, 512))
    # a = np_data[1:30, 2:45]
    # b = slice(*(1,51))
    # crop_ops(np_data)
    # print(3)

    convert_dtype = np.uint8
    crop_range = (64, 64, 32)
    crop_shift = (100, 100, 0)
    overlapping = 1.0
    save_dir = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess'
    build_tmh_crop_data(crop_range, crop_shift, convert_dtype, save_dir, overlapping)
    pass