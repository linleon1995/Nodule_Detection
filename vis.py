import os
import cv2
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import cv2_imshow, calculate_malignancy, segment_lung, mask_preprocess
from utils import raw_preprocess, compare_result, compare_result_enlarge, time_record
# from convert_to_coco_structure import lidc_to_datacatlog_valid
import logging
from sklearn.metrics import confusion_matrix
import time
import pylidc as pl
import pandas as pd
from tqdm import tqdm
from volume_eval import volumetric_data_eval
logging.basicConfig(level=logging.INFO)

    

# def save_mask_in_3d_interface(vol_generator, save_path1, save_path2):
#     volume_generator = vol_generator(cfg.FULL_DATA_PATH, subset_indices=cfg.SUBSET_INDICES, case_indices=cfg.CASE_INDICES,
#                                      only_nodule_slices=cfg.ONLY_NODULES)
#     for vol_idx, (vol, mask_vol, infos) in enumerate(volume_generator):
#         pid, scan_idx = infos['pid'], infos['scan_idx']
#         mask_vol = np.int32(mask_vol)
#         if vol_idx > 9:
#             if np.sum(mask_vol==0) == mask_vol.size:
#                 print('No mask')
#                 continue

#             save_mask_in_3d(mask_vol, save_path1, save_path2)


def save_mask_in_3d(volume, save_path1, save_path2):
    if np.sum(volume==0) == volume.size:
        print('No mask')
    else:
        plot_volume_in_mesh(volume, 0, save_path1)
        volume = volumetric_data_eval.volume_preprocess(volume, connectivity=26, area_threshold=30)
        print(np.unique(volume))
        volume_list = [np.int32(volume==label) for label in np.unique(volume)[1:]]
        plot_volume_in_mesh(volume_list, 0, save_path2)


def show_mask_in_2d(cfg, vol_generator):
    volume_generator = vol_generator(cfg.FULL_DATA_PATH, subset_indices=cfg.SUBSET_INDICES, case_indices=cfg.CASE_INDICES,
                                     only_nodule_slices=cfg.ONLY_NODULES)
    fig, ax = plt.subplots(1, 1, figsize=(4,4))
    for vol_idx, (vol, mask_vol, infos) in enumerate(volume_generator):
        pid, scan_idx = infos['pid'], infos['scan_idx']
        mask_vol = np.int32(mask_vol)
        if vol_idx in [0, 2, 3, 7, 10 ,11]:
            if np.sum(mask_vol==0) == mask_vol.size:
                print('No mask')
                continue
            mask_vol2 = volumetric_data_eval.volume_preprocess(mask_vol, connectivity=26, area_threshold=30)
            def plot_func(volume, name):
                zs, ys, xs = np.where(volume)
                # min_nonzero_slice, max_nonzero_slice = np.min(zs), np.max(zs)
                zs = np.unique(zs)
                for slice_idx in zs:
                    ax.imshow(volume[slice_idx], vmin=0, vmax=5)
                    fig.savefig(os.path.join(cfg.SAVE_PATH, '2d_mask', f'{name}-{vol_idx:03d}-{slice_idx:03d}.png'))

            plot_func(mask_vol, 'raw')
            plot_func(mask_vol2, 'preprocess')


def plot_volume_in_mesh(volume_geroup, threshold=-300, save_path=None): 
    if not isinstance(volume_geroup, list):
        volume_geroup = [volume_geroup]

    # TODO: fix limited colors
    # from itertools import combinations
 
    # # Get all combinations of [1, 2, 3]
    # # and length 2
    # comb = combinations([1, 2, 3], 2)
    
    # # Print the obtained combinations
    # for i in list(comb):
    #     print (i)
    colors = [[0.5, 0.5, 1], [0.5, 1, 0.5], [1, 0.5, 0.5], [0.5, 1, 1], [1, 1, 0.5], [1, 0.5, 1],
              [0.1, 0.7, 1], [0.7, 1, 0.1], [1, 0.7, 0.1], [0.1, 0.7, 0.7], [0.7, 0.7, 0.1], [0.7, 0.1, 0.7]]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for vol_idx, vol in enumerate(volume_geroup):
        p = vol.transpose(2,1,0)
        verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
        mesh = Poly3DCollection(verts[faces], alpha=0.3)
        face_color = colors[vol_idx]
        # face_color = np.random.rand(3)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_3d(image, threshold=-300): 
    p = image.transpose(2,1,0)
    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.3)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()


def save_mask(img, mask, pred, num_class, save_path, save_name='img', mask_or_pred_exist=True):
    if mask_or_pred_exist:
        condition = (np.sum(mask)>0 or np.sum(pred)>0)
    else:
        condition = True
        
    if condition:
        sub_save_path = save_path
        if not os.path.isdir(sub_save_path):
            os.makedirs(sub_save_path)

        fig1, _ = compare_result(img, mask, pred, show_mask_size=True, alpha=0.2, vmin=0, vmax=num_class-1)
        fig1.savefig(os.path.join(sub_save_path, f'{save_name}.png'))
        # fig1.tight_layout()
        plt.close(fig1)

        fig2, _ = compare_result_enlarge(img, mask, pred, show_mask_size=False, alpha=0.2, vmin=0, vmax=num_class-1)
        if fig2 is not None:
            fig2.savefig(os.path.join(sub_save_path, f'{save_name}-en.png'))
            # fig2.tight_layout()
            plt.close(fig2)

