import numpy as np


def crops_to_volume(crops, slices, volume_shape):
    volume = np.zeros(volume_shape)
    for crop, slice in zip(crops, slices):
        volume[slice[0]][:,slice[1]][:,:,slice[2]] = crop
    return volume


# def crops_to_volume(crops, slices, volume_shape):
#     volume, total_times = np.zeros(volume_shape), np.zeros(volume_shape)
#     for crop, slice in zip(crops, slices):
#         temp_volume, times = np.zeros(volume_shape), np.zeros(volume_shape)
#         # temp_volume[*slice] = crop
#         # TODO:
#         temp_volume[slice[0]][:,slice[1]][:,:,slice[2]] = crop
#         times[slice[0]][:,slice[1]][:,:,slice[2]] = 1

#         total_times += times
#         volume += temp_volume
#     print(np.min(volume), np.max(volume))
#     volume /= total_times
#     print(np.min(volume), np.max(volume))
#     print(np.min(total_times), np.max(total_times))
#     return volume