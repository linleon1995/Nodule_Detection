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


# def crops_to_volume(crops, slices, volume_shape):
#     volume = np.zeros(volume_shape)
#     for crop, slice in zip(crops, slices):
#         volume[slice[0]][:,slice[1]][:,:,slice[2]] = crop
#     return volume


def crops_to_volume(crops, slices, volume_shape, reweight=True):
    volume, total_times = np.zeros(volume_shape), np.zeros(volume_shape)
    num_sample = crops.shape[0]
    if reweight:
        importance_map = _get_gaussian(patch_size=(64,64,32), sigma_scale=0.25)
        # importance_maps = np.tile(importance_map[np.newaxis, np.newaxis], (num_sample,1,1,1,1))
        importance_maps = importance_map[np.newaxis, np.newaxis]

        # import matplotlib.pyplot as plt
        # plt.imshow(importance_map[...,16])
        # plt.show()

        crops *= importance_maps
    else:
        importance_map = 1
        
    for crop, crop_slice in zip(crops, slices):
        temp_volume, times = np.zeros(volume_shape), np.zeros(volume_shape)
        crop = crop[0]
        
        # TODO:
        temp_volume[crop_slice[0], crop_slice[1], crop_slice[2]] = crop
        times[crop_slice[0], crop_slice[1], crop_slice[2]] = importance_map
        total_times += times
        volume += temp_volume
    # print(np.min(volume), np.max(volume))
    total_times = np.clip(total_times, 1, None)
    volume /= total_times
    print(f'Value [{np.min(volume)}, {np.max(volume)}]')
    print(f'Times [{np.min(total_times)}, {np.max(total_times)}]')
    return volume