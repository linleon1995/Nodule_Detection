import numpy as np

from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter
from skimage import measure, morphology
from sklearn.cluster import KMeans
from utils.utils import cv2_imshow


def remove_unusual_nodule_by_ratio(pred_vol_individual, lung_mask_vol, threshold=0.019):
    pred_vol = np.where(pred_vol_individual>0, 1, 0)
    pred_pxiel_sum = np.sum(pred_vol, axis=(1,2))
    lung_mask_pxiel_sum = np.sum(lung_mask_vol, axis=(1,2))
    ratio = pred_pxiel_sum / lung_mask_pxiel_sum
    mask = np.where(ratio<threshold, 1, 0)
    mask = np.reshape(mask, [mask.size, 1, 1])
    return pred_vol_individual * mask


def remove_unusual_nodule_by_lung_size(pred_vol_individual, lung_mask_vol, threshold=0.5):
    lung_mask_pxiel_sum = np.sum(lung_mask_vol, axis=(1,2))
    ratio = lung_mask_pxiel_sum / np.max(lung_mask_pxiel_sum)
    mask = np.where(ratio>=threshold, 1, 0)
    mask = np.reshape(mask, [mask.size, 1, 1])
    return pred_vol_individual * mask


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


        