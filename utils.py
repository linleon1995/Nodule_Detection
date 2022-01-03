import os
import cv2
import numpy as np
import site_path
import pylidc as pl
from pylidc.utils import consensus
from statistics import median_high
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter
from skimage import measure, morphology
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
from modules.data import dataset_utils



def calculate_malignancy(nodule):
    # Calculate the malignancy of a nodule with the annotations made by 4 doctors. Return median high of the annotated cancer, True or False label for cancer
    # if median high is above 3, we return a label True for cancer
    # if it is below 3, we return a label False for non-cancer
    # if it is 3, we return ambiguous
    list_of_malignancy =[]
    for annotation in nodule:
        list_of_malignancy.append(annotation.malignancy)

    malignancy = median_high(list_of_malignancy)
    if  malignancy > 3:
        return malignancy, True
    elif malignancy < 3:
        return malignancy, False
    else:
        return malignancy, 'Ambiguous'


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
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    # mask consists of 1 and 0. Thus by mutliplying with the orginial image, sections with 1 will remain
    return mask*img





def raw_preprocess(img, lung_segment=False, norm=True, change_channel=True, output_dtype=np.int32):
    if lung_segment:
        assert img.ndim == 2
        img = segment_lung(img)
        img[img==-0] = 0

    if norm:
        dtype = img.dtype
        img = np.float32(img) # avoid overflow
        if np.max(img)==np.min(img):
            img = np.zeros_like(img)
        else:
            img = 255*((img-np.min(img))/(np.max(img)-np.min(img)))
        img = np.array(img, dtype)
    
    if change_channel:
        img = np.tile(img[...,np.newaxis], np.append(np.ones(img.ndim, dtype=np.int32), 3))

    img = output_dtype(img)
    return img


def mask_preprocess(mask, ignore_malignancy=True, output_dtype=np.int32):
    # assert mask.ndim == 2
    if ignore_malignancy:
        mask = np.where(mask>=1, 1, 0)
    mask = output_dtype(mask)
    return mask


def compare_result(image, label, pred, **imshow_params):
    if 'alpha' not in imshow_params: imshow_params['alpha'] = 0.2
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(image)
    ax[0].imshow(label, **imshow_params)
    ax[1].imshow(image)
    ax[1].imshow(pred, **imshow_params)
    ax[0].set_title('Label')
    ax[1].set_title('Prediction')
    return fig, ax


def compare_result_enlarge(image, label, pred, **imshow_params):
    crop_range = 30
    if np.sum(label) > 0:
        item = label
    else:
        if np.sum(pred) > 0:
            item = pred
        else:
            item = None
    
    fig, ax = None, None
    if item is not None:
        if image.ndim == 2:
            image = raw_preprocess(image, lung_segment=False, norm=False)
        image = np.uint8(image)
        h, w, c = image.shape
        ys, xs = np.where(item)
        x1, x2 = max(0, min(xs)-crop_range), min(max(xs)+crop_range, min(h,w))
        y1, y2 = max(0, min(ys)-crop_range), min(max(ys)+crop_range, min(h,w))
        bbox_size = np.max([np.abs(x1-x2), np.abs(y1-y2)])

        image = cv2.resize(image[y1:y1+bbox_size, x1:x1+bbox_size], (w, h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label[y1:y1+bbox_size, x1:x1+bbox_size], (w, h), interpolation=cv2.INTER_NEAREST)
        pred = cv2.resize(pred[y1:y1+bbox_size, x1:x1+bbox_size], (w, h), interpolation=cv2.INTER_NEAREST)

        fig, ax = compare_result(image, label, pred, **imshow_params)
    return fig, ax


def cv2_imshow(img):
    # pass
    cv2.imshow('My Image', img)
    cv2.imwrite('sample.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convert_npy_to_png(src, dst, src_format, dst_format):
    src_files = dataset_utils.get_files(src, src_format)
    if not os.path.isdir(dst):
        os.makedirs(dst)
    for idx, f in enumerate(src_files):
        print(idx, f)
        img = np.load(f)
        # if np.sum(img==1):
        #     print(3)
        #     import matplotlib.pyplot as plt
        #     plt.imshow(img)
        #     plt.show()

        # TODO:
        # if img.dtype == bool:
        #     img = np.uint8(img)
        # else:
        #     img = np.uint8(255*((img-np.min(img))/(np.max(img)-np.min(img))))
        new_f = os.path.join(os.path.split(f)[0].replace(src, dst), os.path.split(f)[1].replace(src_format, dst_format))
        if not os.path.isdir(os.path.split(new_f)[0]):
            os.makedirs(os.path.split(new_f)[0])
        cv2.imwrite(new_f, img)

