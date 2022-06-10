import numpy as np
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate
import math
import time
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt


def image_3d_cls_transform(img, mask=None):
    input_dim = img.ndim
    assert (input_dim==3 or input_dim==4), 'Incorret dimension for 3D image transformation'
    if input_dim == 3:
        img = img[None]
    # img, mask = scale_3d(img, mask)
    img, mask = crop_and_scale_3d(img, mask)
    img, mask = rotate_3d(img, mask)
    img, mask = flip_3d(img, mask)
    img, mask = swap_3d(img, mask)
    if input_dim == 3:
        img = img[0]
    # for i in range(0, img.shape[0], 2):
    #     plt.imshow(img[i], 'gray')
    #     plt.show()
    return img, mask


def flip_3d(img, mask=None):
    flipid = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2-1
    img = np.ascontiguousarray(img[:, ::flipid[0], ::flipid[1], ::flipid[2]])
    if mask is not None:
        mask = np.ascontiguousarray(mask[::flipid[0],::flipid[1],::flipid[2]])

    # for ax in range(3):
    #     if flipid[ax]==-1:
    #         target[ax] = np.array(img.shape[ax+1])-target[ax]
    return img, mask


def swap_3d(img, mask=None):
    axisorder = np.random.permutation(2) + 2
    img = np.transpose(img, np.concatenate([[0, 1], axisorder]))
    if mask is not None:
        mask = np.transpose(mask, np.concatenate([[0, 1], axisorder]))
    # if img.shape[1]==img.shape[2] and img.shape[1]==img.shape[3]:
    #     axisorder = np.random.permutation(3)
    #     img = np.transpose(img, np.concatenate([[0],axisorder+1]))
    #     coord = np.transpose(coord, np.concatenate([[0],axisorder+1]))
    return img, mask


def rotate_3d(img, mask, angle_range=(-30, 30)):
    # angle = np.random.rand()*180
    angle = (np.random.rand()*2-1)*30
    # angle = np.random.uniform(*angle_range)
    axes = tuple(np.random.choice(3, 2, replace=False)+1)
    # angle = 0.0
    # axes = (1, 2)
    img = rotate(img, angle,axes=axes, reshape=False, mode='reflect')
    # TODO: Not gaunrantee rotate working on mask (need to check value, dimension for multi class semantic seg label)
    # TODO: Better implementation of angle range
    if mask is not None:
        mask = rotate(mask, angle, axes=axes, reshape=False)
    return img, mask


def scale_3d(img, mask, crop_size, scale, pad_value):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = zoom(img, [scale, scale, scale], order=1)
        mask = zoom(mask, [scale, scale, scale], order=1)
    newpad = crop_size[0] - img.shape[1:][0]
    if newpad<0:
        img = img[:,:-newpad,:-newpad,:-newpad]
        mask = mask[:-newpad,:-newpad,:-newpad]
    elif newpad>0:
        pad2 = [[0, 0], [0, newpad], [0, newpad], [0, newpad]]
        img = np.pad(img, pad2, 'constant', constant_values=pad_value)
        mask = np.pad(mask, pad2[1:], 'constant', constant_values=0)

    # for i in range(4):
    #     target[i] = target[i]*scale
    return img, mask

def crop_and_scale_3d(img, mask=None, ratio=0.8):
    # TODO: channel dimension issue, cropping operation should able to map in all the channel if exist
    # e.g., [512, 512, 3] --> [64,64,3]
    img_shape = img.shape
    if img.ndim == 4:
        img_shape = img_shape[1:]

    new_img_shape = np.int32(np.array(img_shape, 'float')*ratio)
    max_start_point = img_shape - new_img_shape
    start_point = np.random.randint(max_start_point)
    slice_bbox = [slice(start, start+new_img_shape[idx]) for idx, start in enumerate(start_point)]
    img = img[:, slice_bbox[0], slice_bbox[1], slice_bbox[2]]
    if mask is not None:
        mask = mask[:, slice_bbox[0], slice_bbox[1], slice_bbox[2]]

    zoom_factor = np.concatenate((np.ones(1), np.float32(img_shape/new_img_shape)))
    img = zoom(img, zoom_factor)
    # TODO: again, the correctness of zoom function use on mask is not promissing
    if mask is not None:
        mask = zoom(mask, zoom_factor)
    return img, mask
