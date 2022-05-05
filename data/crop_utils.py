import numpy as np
from scipy.ndimage.filters import gaussian_filter


def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def crops_to_volume(crops, slices, volume_shape, reweight, reweight_sigma):
    if reweight:
        importance_map = _get_gaussian(patch_size=(64,64,32), sigma_scale=reweight_sigma)
        importance_maps = importance_map[np.newaxis, np.newaxis]
        crops *= importance_maps
    else:
        importance_map = 1
        
    volume, times = np.zeros(volume_shape), np.zeros(volume_shape)
    crops = np.squeeze(crops)
    for crop, crop_slice in zip(crops, slices):
        # TODO:
        if np.sum(volume[crop_slice]):
            volume[crop_slice] = volume[crop_slice] + crop
        else:
            volume[crop_slice] = crop

        if np.sum(times[crop_slice]):
            times[crop_slice] = times[crop_slice] + importance_map
        else:
            times[crop_slice] = importance_map

    times = np.clip(times, 1, None)
    volume /= times
    print(f'Value [{np.min(volume)}, {np.max(volume)}]')
    print(f'Times [{np.min(times)}, {np.max(times)}]')
    return volume


def volume_to_crops(volume, crop_size, crop_shift, convert_dtype, overlapping):
    crop_op = CropVolumeOP(crop_size, crop_shift, convert_dtype, overlapping)

class CropVolumeOP():
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
        
        slice_indices = np.indices(self.crop_size)
        for dim, slice_range in enumerate(crop_slices):
            for dim_shift in slice_range:
                slice_indices[dim]
                shift = np.zeros(3, dtype=np.int32)
                shift[dim] = dim_shift
                shift = np.reshape(shift, (shift.size, 1, 1, 1))
                shift_slice_indices = slice_indices + shift
                # crop_slice = [slice(*r) for r in slice_range]
                crop = volume[shift_slice_indices[0], shift_slice_indices[1], shift_slice_indices[2]]
                crop_data.append({'slice': shift_slice_indices, 'data':crop})
        return crop_data

    def get_crop_slice(self, volume_shape):
        slices = []
        for dim_idx, length in enumerate(volume_shape):
            dim_slice = self.simple_slice(length, self.crop_shift[dim_idx], self.crop_size[dim_idx], self.overlapping)
            slices.append(dim_slice)

        # # TODO: for arbitrary dimensions
        # slice_comb = []
        # for h in slices[0]:
        #     for w in slices[1]:
        #         for c in slices[2]: 
        #             slice_comb.append([h, w, c])
        return slices

    @staticmethod
    def simple_slice(length, ignore_range, crop_length, overlapping):
        step = int(crop_length*overlapping)
        slices = np.arange(ignore_range+crop_length//2, length-ignore_range-crop_length//2, step)
        return slices

        
