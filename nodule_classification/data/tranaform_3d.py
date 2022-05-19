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
    img, mask = rotate_3d(img, mask)
    img, mask = flip_3d(img, mask)
    img, mask = swap_3d(img, mask)
    if input_dim == 3:
        img = img[0]
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
    if img.shape[1]==img.shape[2] and img.shape[1]==img.shape[3]:
        axisorder = np.random.permutation(3)
        img = np.transpose(img, np.concatenate([[0],axisorder+1]))
        coord = np.transpose(coord, np.concatenate([[0],axisorder+1]))
        # target[:3] = target[:3][axisorder]
        # bboxes[:,:3] = bboxes[:,:3][:,axisorder]
    return img, mask


def rotate_3d(img, mask, angle_range=(-30, 30)):
    # angle = np.random.rand()*180
    angle = (np.random.rand()*2-1)*30
    # angle = np.random.uniform(*angle_range)
    img = rotate(img, angle,axes=(2,3),reshape=False)
    # TODO: Not gaunrantee rotate working on mask (need to check value, dimension for multi class semantic seg label)
    # TODO: Better implementation of angle range
    if mask is not None:
        mask = rotate(mask, angle, axes=(1,2), reshape=False)
    # plt.imshow(img[10])
    # plt.show()
    
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

def crop_3d():
    pass

