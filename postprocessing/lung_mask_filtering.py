import numpy as np
import os
import cv2

from medpy.filter.smoothing import anisotropic_diffusion
from pyrsistent import v
from scipy.ndimage import median_filter
from skimage import measure, morphology
from sklearn.cluster import KMeans

from data.data_utils import get_files


# TODO: add time recording
class FalsePositiveReducer():
    def __init__(self, _1SR, RUNLS, LMF, slice_threshold=1, lung_size_threshold=0.4):
        self._1SR = _1SR
        self.RUNLS = RUNLS
        self.LMF = LMF
        self.slice_threshold = slice_threshold
        self.lung_size_threshold = lung_size_threshold
    
    def __call__(self, pred_study, raw_vol, lung_mask_path, pid, lung_mask_vol=None):
        # TODO: every removal results is based on last process result, check is this
        # the behavior we want.
        if self._1SR:
            pred_study = under_slice_removal(pred_study, self.slice_threshold)

        if self.LMF or self.RUNLS:
            if lung_mask_vol is not None:
                lung_mask_case_path = os.path.join(lung_mask_path, pid)
                if not os.path.isdir(lung_mask_case_path):
                    os.makedirs(lung_mask_case_path)
                    lung_mask_vol = get_lung_mask(raw_vol)
                    for lung_mask_idx, lung_mask in enumerate(lung_mask_vol):
                        cv2.imwrite(os.path.join(lung_mask_case_path, f'{pid}-{lung_mask_idx:03d}.png'), 255*lung_mask)
                else:
                    lung_mask_files = get_files(lung_mask_case_path, 'png')
                    lung_mask_vol = np.zeros_like(pred_study.category_volume)
                    for lung_mask_idx, lung_mask in enumerate(lung_mask_files): 
                        lung_mask_vol[lung_mask_idx] = cv2.imread(lung_mask)[...,0]
                    lung_mask_vol = lung_mask_vol / 255
            lung_mask_vol = np.uint8(lung_mask_vol)

            if self.RUNLS:
                pred_study = remove_unusual_nodule_by_lung_size(
                    pred_study, lung_mask_vol, min_lung_ration=self.lung_size_threshold)
             
            if self.LMF:
                pred_study = lung_masking(pred_study, lung_mask_vol)
            return pred_study

                
def under_slice_removal(pred_study, slice_threshold=1):
    remove_nodule_ids = []
    for nodule_id, nodule in pred_study.nodule_instances.items():
        max_z = nodule.nodule_range['index']['max']
        min_z = nodule.nodule_range['index']['min']
        if max_z-min_z < slice_threshold:
            remove_nodule_ids.append(nodule_id)

    pred_study.record_nodule_removal(name='_1SR', nodules_ids=remove_nodule_ids)
    return pred_study


def remove_unusual_nodule_by_ratio(pred_study, lung_mask_vol, threshold=0.019):
    pred_vol_category = pred_study.category_volume
    pred_vol = np.where(pred_vol_category>0, 1, 0)
    pred_pxiel_sum = np.sum(pred_vol, axis=(1,2))
    lung_mask_pxiel_sum = np.sum(lung_mask_vol, axis=(1,2))
    ratio = pred_pxiel_sum / lung_mask_pxiel_sum
    mask = np.where(ratio<threshold, 1, 0)
    mask = np.reshape(mask, [mask.size, 1, 1])
    pred_vol_category = pred_vol_category * mask

    remove_nodule_ids = get_nodule_id(pred_vol_category)
    pred_study.record_nodule_removal(name='RUNR', nodules_ids=remove_nodule_ids)
    return pred_vol_category


def remove_unusual_nodule_by_lung_size(pred_study, lung_mask_vol, min_lung_ration=0.5):
    lung_mask_pxiel_sum = np.sum(lung_mask_vol, axis=(1,2))
    ratio = lung_mask_pxiel_sum / np.max(lung_mask_pxiel_sum)
    remove_mask = np.where(ratio>=min_lung_ration, 0, 1)
    remove_mask = np.reshape(remove_mask, [remove_mask.size, 1, 1])

    pred_vol_category = pred_study.category_volume * remove_mask

    remove_nodule_ids = get_nodule_id(pred_vol_category)
    pred_study.record_nodule_removal(name='RUNLS', nodules_ids=remove_nodule_ids)
    return pred_study


def lung_masking(pred_study, lung_mask_vol):
    pred_vol_category = pred_study.category_volume * (np.ones_like(lung_mask_vol)-lung_mask_vol)
    remove_nodule_ids = get_nodule_id(pred_vol_category)
    pred_study.record_nodule_removal('LMF', nodules_ids=remove_nodule_ids)
    return pred_study


def get_nodule_id(pred_vol_category):
    return [pred_nodule_id for pred_nodule_id in np.unique(pred_vol_category)[1:]]


def get_lung_mask(input_vol=None, lung_mask_path=None):
    assert input_vol is not None or lung_mask_path is not None, \
    'Either load preprocess lung mask or calculate from input image'

    if lung_mask_path is not None:
        lung_mask_vol = np.load(lung_mask_path)
    else:
        lung_mask_vol = np.zeros_like(input_vol)
        for img_idx, img in enumerate(input_vol):
            lung_mask_vol[img_idx] = segment_lung(img)
    return lung_mask_vol


def segment_lung(img):
    #function sourced from https://www.kaggle.com/c/data-science-bowl-2017#tutorial
    """
    This segments the Lung Image(Don't get confused with lung nodule segmentation)
    """
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    
    middle = img[100:400,100:400] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    #remove the underflow bins
    img[img==max]=mean
    img[img==min]=mean
    
    #apply median filter
    img= median_filter(img,size=3)
    #apply anistropic non-linear diffusion filter- This removes noise without blurring the nodule boundary
    img= anisotropic_diffusion(img)
    
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
    eroded = morphology.erosion(thresh_img,np.ones([4,4]))
    dilation = morphology.dilation(eroded,np.ones([10,10]))
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
            good_labels.append(prop.label)
    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0
    #
    #  The mask here is the mask for the lungs--not the nodes
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    # mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    mask = morphology.dilation(mask,np.ones([30,30])) # one last dilation
    # mask consists of 1 and 0. Thus by mutliplying with the orginial image, sections with 1 will remain
    return mask


        