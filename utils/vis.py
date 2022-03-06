import os
import cv2
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils.utils import cv2_imshow, calculate_malignancy, segment_lung, mask_preprocess
from utils.utils import raw_preprocess, compare_result, compare_result_enlarge, time_record
# from convert_to_coco_structure import lidc_to_datacatlog_valid
import logging
from sklearn.metrics import confusion_matrix
import time
import pylidc as pl
import pandas as pd
from tqdm import tqdm
import cv2
import cc3d
from utils.volume_eval import volumetric_data_eval
logging.basicConfig(level=logging.INFO)
CROP_HEIGHT, CROP_WIDTH = 128, 128
    

def crop_img(image, crop_parameters, keep_size=True):
    if not isinstance(crop_parameters, list):
        crop_parameters = [crop_parameters]
    
    height, width = image.shape[:2]
    crop_images = []
    for crop_parameter in crop_parameters:
        crop_center, crop_height, crop_width = crop_parameter['center'], crop_parameter['height'], crop_parameter['width']
        y_start, y_end = np.clip(crop_center['y']-crop_height//2, 0, height-1), np.clip(crop_center['y']+crop_height//2, 0, height-1)
        x_start, x_end = np.clip(crop_center['x']-crop_width//2, 0, width-1), np.clip(crop_center['x']+crop_width//2, 0, width-1)

        crop_image = image[y_start:y_end, x_start:x_end]
        if keep_size:
            crop_image = cv2.resize(crop_image, dsize=(width, height))
        crop_images.append(crop_image)
    return crop_images
        

def visualize(input_vol, pred_vol, target_vol, pred_nodule_info):
    # pred_vol = cc3d.connected_components(pred_vol, connectivity=26)
    target_vol = cc3d.connected_components(target_vol, connectivity=26)

    pred_category = np.unique(pred_vol)[1:]
    zs, ys, xs = np.where(pred_vol)
    pred_zs = np.unique(zs)
    zs, ys, xs = np.where(target_vol)
    target_zs = np.unique(zs)
    total_zs = np.unique(np.concatenate((pred_zs, target_zs)))
    
    # TODO: convert any color map <-- write this as a function
    color_map = cm.tab20
    bgr_colors = [np.uint8(255*np.array(color_map(idx))[:3])[::-1] for idx in range(20)]
    target_colors, pred_colors = bgr_colors[4:5], bgr_colors[:4]+bgr_colors[5:14]+bgr_colors[16:]
    draw_vol = input_vol.copy()
    center_shift = 20

    # Find prediction contours
    total_contours = {}
    for nodule_id in pred_category:
        nodule_vol = np.uint8(pred_vol==nodule_id)
        # nodule_vol = np.tile(nodule_vol[...,np.newaxis], (1,1,1,3))
        zs, ys, xs = np.where(nodule_vol)
        total_contours[nodule_id] = {}
        for z_idx in range(np.min(zs), np.max(zs)+1):
            pred = nodule_vol[z_idx]
            contours, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            total_contours[nodule_id][z_idx] = contours

    # Draw prediction contours
    for nodule_id in total_contours:
        contour_pair = total_contours[nodule_id]
        color = pred_colors[nodule_id%len(pred_colors)].tolist()
        prob = pred_nodule_info[nodule_id]['Nodule_pred_prob'][1]
        for slice_idx in contour_pair:
            contours = contour_pair[slice_idx]
            draw_img = draw_vol[slice_idx]
            contour_center = np.int32(np.mean(contours[0], axis=0)[0]) + center_shift
            # TODO: shifting for any image size
            contour_center = np.clip(contour_center, 0, 512)

            for contour_idx in range(len(contours)):
                draw_img = cv2.drawContours(draw_img, contours, contour_idx, color, 1)
            
            # probability text
            draw_img = cv2.putText(draw_img, f'{prob:.4f}', contour_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            draw_vol[slice_idx] = draw_img


    # Find and draw target contours
    target_vol = np.uint8(target_vol)
    for z_idx in target_zs:
        target = target_vol[z_idx]
        contours, hierarchy = cv2.findContours(target, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        draw_vol[z_idx] = cv2.drawContours(draw_vol[z_idx], contours, -1, target_colors[0].tolist(), 1)

    crop_draw_imgs = {}
    candidate_vol = cc3d.connected_components(np.where(pred_vol+target_vol>0, 1, 0), connectivity=26)
    for z_idx in total_zs:
        candidate_slice = candidate_vol[z_idx]
        category = np.unique(candidate_slice).tolist()
        category.remove(0)
        for ccl_id in category:
            ys, xs = np.where(candidate_slice==ccl_id)
            center = {'y': np.int16(np.mean(ys)), 'x': np.int16(np.mean(xs))}
            # TODO: parameters
            crop_parameter = {'center': center, 'height': CROP_HEIGHT, 'width': CROP_WIDTH}
            crops = crop_img(draw_vol[z_idx], crop_parameter)
        crop_draw_imgs[z_idx] = crops
        # plt.imshow(draw_vol[z_idx])
        # plt.show()

    return draw_vol, total_zs, crop_draw_imgs


# class ObjectVisualizer():
#     def __init__(self):
#         self.visulizer = self.get_visualizer()

#     def get_visualizer(self):
#         return visualizer

#     def visualize(self, input_vol, pred_vol, mask_vol):
#         pred_category = np.unique(pred_vol)[1:]
#         zs, ys, xs = np.where(pred_vol)
#         total_zs = np.unique(zs)
#         color_map = cm.tab20
#         colors = [np.uint8(255*np.array(color)[:3]) for color in color_map]
#         target_colors, pred_colors = colors[:2], colors[2:]
#         draw_vol = input_vol.copy()

#         total_contours = {}
#         for label in pred_category:
#             nodule_vol = pred_vol==label
#             zs, ys, xs = np.where(nodule_vol)
#             total_contours[label] = {}
#             for z_idx in range(np.min(zs), np.max(zs)+1):
#                 pred = nodule_vol[z_idx]
#                 _, contours, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#                 total_contours[label][z_idx] = contours

#         for nodule_id in total_contours:
#             contour_pair = total_contours[nodule_id]
#             color = pred_colors[nodule_id]
#             for slice_idx in contour_pair:
#                 contours = contour_pair[slice_idx]
#                 draw_img = draw_vol[slice_idx]
#                 for contour_idx in range(len(contours)):
#                     draw_img = cv2.drawContours(draw_img, contours, contour_idx, pred_colors[contour_idx], 3)
#                     draw_vol[slice_idx] = draw_img

#         for z_idx in total_zs:
#             plt.imshow(draw_vol[z_idx])
#             plt.show()

#         # return draw_vol


def save_mask_in_3d_2(volume, save_path1, save_path2, crop_range=None):
    
    if np.sum(volume==0) == volume.size:
        print('No mask')
    else:
        plot_volume_in_mesh(volume, 0, save_path1)
        volume, _ = volumetric_data_eval.volume_preprocess(volume, connectivity=26, area_threshold=30)
        if crop_range is not None:
            volume = volume[crop_range['z'][0]:crop_range['z'][1], crop_range['y'][0]:crop_range['y'][1], crop_range['x'][0]:crop_range['x'][1]]

        volume_list = [np.int32(volume==label) for label in np.unique(volume)[1:]]
        plot_volume_in_mesh(volume_list, 0, save_path2)


def save_mask_in_3d(volume, save_path1, save_path2, enlarge=True):
    if enlarge:
        zs, ys, xs = np.where(volume)
        crop_range = {'z': (np.min(zs), np.max(zs)), 'y': (np.min(ys), np.max(ys)), 'x': (np.min(xs), np.max(xs))}
        volume = volume[crop_range['z'][0]:crop_range['z'][1], crop_range['y'][0]:crop_range['y'][1], crop_range['x'][0]:crop_range['x'][1]]

    if np.sum(volume==0) == volume.size:
        print('No mask')
    else:
        plot_volume_in_mesh(volume, 0, save_path1)
        volume = volumetric_data_eval.volume_preprocess(volume, connectivity=26, area_threshold=30)
        # print(np.unique(volume))
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
    # TODO: change color map
    colors = [[0.5, 0.5, 1], [0.5, 1, 0.5], [1, 0.5, 0.5], [0.5, 1, 1], [1, 1, 0.5], [1, 0.5, 1],
              [0.1, 0.7, 1], [0.7, 1, 0.1], [1, 0.7, 0.1], [0.1, 0.7, 0.7], [0.7, 0.7, 0.1], [0.7, 0.1, 0.7]]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for vol_idx, vol in enumerate(volume_geroup):
        p = vol.transpose(2,1,0)
        verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
        mesh = Poly3DCollection(verts[faces], alpha=0.3)
        face_color = colors[vol_idx%len(colors)]
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


def save_mask(img, mask, pred, num_class, save_path, save_name='img', saving_condition=True):
    save_name = save_name.split('.')[-1]
    if saving_condition:
        condition = (np.sum(mask)>0 or np.sum(pred)>0)
    else:
        condition = True
        
    if condition:
        origin_save_path = os.path.join(save_path, 'origin')
        if not os.path.isdir(origin_save_path):
            os.makedirs(origin_save_path)
        fig1, _ = compare_result(img, mask, pred, show_mask_size=True, alpha=0.2, vmin=0, vmax=num_class-1)
        fig1.savefig(os.path.join(origin_save_path, f'{save_name}.png'))
        plt.close(fig1)

        enlarge_save_path = os.path.join(save_path, 'enlarge')
        if not os.path.isdir(enlarge_save_path):
            os.makedirs(enlarge_save_path)
        fig2, _ = compare_result_enlarge(img, mask, pred, show_mask_size=False, alpha=0.2, vmin=0, vmax=num_class-1)
        if fig2 is not None:
            fig2.savefig(os.path.join(enlarge_save_path, f'{save_name}-en.png'))
            plt.close(fig2)


class visualizer():
    def __init__(self, save_path):
        self.save_path = save_path

    @staticmethod
    def visualize_segmentation_in_2d(image, mask, pred, num_class, save_path, save_name='img', mask_or_pred_exist=True):
        if mask_or_pred_exist:
            condition = (np.sum(mask)>0 or np.sum(pred)>0)
        else:
            condition = True
            
        if condition:
            sub_save_path = save_path
            if not os.path.isdir(sub_save_path):
                os.makedirs(sub_save_path)

            fig1, _ = compare_result(image, mask, pred, show_mask_size=True, alpha=0.2, vmin=0, vmax=num_class-1)
            fig1.savefig(os.path.join(sub_save_path, f'{save_name}.png'))
            plt.close(fig1)

            fig2, _ = compare_result_enlarge(image, mask, pred, show_mask_size=False, alpha=0.2, vmin=0, vmax=num_class-1)
            if fig2 is not None:
                fig2.savefig(os.path.join(sub_save_path, f'{save_name}-en.png'))
                plt.close(fig2)

    @staticmethod
    def visualize_segmentation_in_3d(cls):
        pass

    def visulize_sample(self, ):
        self.visualize_segmentation_in_2d()
        self.visualize_segmentation_in_3d()

