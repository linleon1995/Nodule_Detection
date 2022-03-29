from distutils.log import info
import os

# TODO: consider overlapping
import numpy as np
from pyrsistent import v
from data.volume_generator import asus_nodule_volume_generator


class CropVolume():
    def __init__(self, crop_size, crop_shift, convert_dtype=np.uint8):
        self.crop_size = crop_size
        self.crop_shift = crop_shift
        self.convert_dtype = convert_dtype

    def __call__(self, volume):
        crop_data = []
        volume = self.convert_dtype(volume)
        crop_slices = self.get_crop_slice(volume.shape)
        
        for slice_range in crop_slices:
            # crop = volume.copy()
            # for dim , crop_slice_dim in enumerate(crop_slice):
            #     crop = np.take(crop, crop_slice_dim, dim)
            crop_slice = [slice(*r) for r in slice_range]
            crop = volume[crop_slice]
            crop_data.append({'slice': slice_range, 'data':crop})
        return crop_data

    def get_crop_slice(self, volume_shape):
        slices = []
        for dim_idx, length in enumerate(volume_shape):
            slices.append(self.simple_slice(length, self.crop_shift[dim_idx], self.crop_size[dim_idx]))

        # TODO: for arbitrary dimensions
        slice_comb = []
        for h in slices[0]:
            for w in slices[1]:
                for c in slices[2]: 
                    slice_comb.append([h, w, c])

        # def comb(total_slices, slices, dim):
        #     if dim == len(volume_shape)+1:
        #         return total_slices
        #     else:
        #         dim += 1
        #         for slice in total_slices:
        #             comb([slice], slices, dim)
        #         # comb(total_slices, slices, dim)
        # total_slices = comb(slices[0], slices[1:], 0)
        return slice_comb

    @staticmethod
    def simple_slice(length, shift, crop_length):
        slices = []
        for start in range(shift, length-shift-crop_length, crop_length):
            # slices.append(slice(start, start+crop_length))
            slices.append((start, start+crop_length))
        return slices


def build_crop_data(volume_generator, cropping_op, save_dir):
    for vol_idx, (raw_vol, vol, mask_vol, infos) in enumerate(volume_generator):
        pid = infos['pid']
        print(f'{save_dir}-{pid}')

        def save_vol(vol, dir_key):
            vol = np.swapaxes(np.swapaxes(vol, 0, 1), 1, 2)
            crop_data_list = cropping_op(vol)
            save_path = os.path.join(save_dir, dir_key, pid)
            os.makedirs(save_path, exist_ok=True)
            for crop_idx, crop_data in enumerate(crop_data_list):
                np.save(os.path.join(save_path, f'{pid}-{crop_idx:03d}.npy'), crop_data['data'])

        save_vol(vol[...,0], dir_key='Image')
        save_vol(mask_vol, dir_key='Mask')


def build_tmh_crop_data(crop_range, crop_shift, save_dir):
    cropping_op = CropVolume(crop_range, crop_shift)
    diranme = 'x'.join([str(s) for s in list(crop_range)])

    # Malignant
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodule\ASUS-Malignant\merge'
    case_pids = [f'1m{idx:04d}' for idx in range(1, 45)]
    volume_generator = asus_nodule_volume_generator(data_path, case_pids=case_pids)
    build_crop_data(volume_generator, cropping_op, os.path.join(save_dir, 'TMH-Malignant', diranme))

    # Benign
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodule\ASUS-Benign\merge'
    case_pids = [f'1B{idx:04d}' for idx in range(1, 26)]
    volume_generator = asus_nodule_volume_generator(data_path, case_pids=case_pids)
    build_crop_data(volume_generator, cropping_op, os.path.join(save_dir, 'TMH-Benign', diranme))




if __name__ == '__main__':
    # crop_ops = CropVolume((32,64,64), (0, 100, 100))
    # np_data = np.zeros((100, 512, 512))
    # a = np_data[1:30, 2:45]
    # b = slice(*(1,51))
    # crop_ops(np_data)
    # print(3)

    crop_range = (64, 64, 32)
    crop_shift = (100, 100, 0)
    save_dir = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess'
    build_tmh_crop_data(crop_range, crop_shift, save_dir)
    pass