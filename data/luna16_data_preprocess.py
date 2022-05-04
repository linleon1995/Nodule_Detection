import numpy as np
import os
import pandas as pd
from torch import short
from tqdm import tqdm
import pylidc as pl
from pylidc.utils import consensus
from statistics import median_high
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from skimage.measure import find_contours

from data.data_utils import get_files, load_itk
from data.volume_generator import luna16_volume_generator

DICOM_DIR = rf'D:\Leon\Datasets\LIDC-IDRI'



def radiologist_consensus(anns, num_radiologist=3, pad=None, ret_masks=True):
    """
    Change confidence consensus to radiologist consensus
    Output consensus mask with at least [num_radiologist] radiologist agree
    """
    bmats = np.array([a.bbox_matrix(pad=pad) for a in anns])
    imin,jmin,kmin = bmats[:,:,0].min(axis=0)
    imax,jmax,kmax = bmats[:,:,1].max(axis=0)

    # consensus_bbox
    cbbox = np.array([[imin,imax],
                      [jmin,jmax],
                      [kmin,kmax]])

    masks = [a.boolean_mask(bbox=cbbox) for a in anns]
    cmask = np.sum(masks, axis=0) >= num_radiologist
    cbbox = tuple(slice(cb[0], cb[1]+1, None) for cb in cbbox)

    if ret_masks:
        return cmask, cbbox, masks
    else:
        return cmask, cbbox


def lidc_consensus():
    lidc_id = 'LIDC-IDRI-0055'
    clevel = 0.25
    annot_idx = 0

    # Query for a scan, and convert it to an array volume.
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == lidc_id).first()
    vol = scan.to_volume()

    # Cluster the annotations for the scan, and grab one.
    nods = scan.cluster_annotations()
    anns = nods[annot_idx]

    # Perform a consensus consolidation and 50% agreement level.
    # We pad the slices to add context for viewing.
    cmask,cbbox,masks = consensus(anns, clevel=clevel,
                                pad=[(20,20), (20,20), (0,0)])

    # Get the central slice of the computed bounding box.
    k = int(0.5*(cbbox[2].stop - cbbox[2].start))

    # Set up the plot.
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.imshow(vol[cbbox][:,:,k], cmap=plt.cm.gray, alpha=0.5)

    # Plot the annotation contours for the kth slice.
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'w']
    for j in range(len(masks)):
        print(masks[j].shape)
        for a in range(6):
            plt.title(f'{j}-{a}')
            plt.imshow(vol[cbbox][:,:,a], cmap=plt.cm.gray)
            # plt.imshow(masks[j][...,a], alpha=0.2)
            plt.imshow(cmask[...,a], alpha=0.2)
            plt.show()
        for c in find_contours(masks[j][:,:,k].astype(float), 0.5):
            label = "Annotation %d" % (j+1)
            print(label)
            plt.plot(c[:,1], c[:,0], colors[j], label=label)

    # Plot the 50% consensus contour for the kth slice.
    for c in find_contours(cmask[:,:,k].astype(float), 0.5):
        plt.plot(c[:,1], c[:,0], '--k', label='50% Consensus')

    ax.axis('off')
    ax.legend()
    plt.tight_layout()
    #plt.savefig("../images/consensus.png", bbox_inches="tight")
    plt.show()


def save_luna16_nodule_npy(input_root, save_root, data_type):
    data_list = get_files(input_root, keys='mhd')
    for idx, path in enumerate(data_list):
        folder, filename = os.path.split(path)
        _, subset = os.path.split(folder)
        pid = filename[:-4]
        print(f'{idx}/{len(data_list)} {pid}')

        raw_vol, vol, mask_vol, infos = luna16_volume_generator.get_data_by_pid(pid)
        save_path = os.path.join(save_root, subset)
        if data_type == 'raw':
            data = np.int32(raw_vol)
        elif data_type == 'mask':
            data = np.uint8(mask_vol)

        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, filename), data)


def save_luna16_nodule_raw_npy(input_root, save_root):
    data_list = get_files(input_root, keys='mhd')
    for idx, path in enumerate(data_list):
        folder, filename = os.path.split(path)
        _, subset = os.path.split(folder)
        pid = filename[:-4]
        print(f'{idx}/{len(data_list)} {pid}')

        raw_vol, vol, mask_vol, infos = luna16_volume_generator.get_data_by_pid(pid)
        mask_vol = np.uint8(mask_vol)
        save_path = os.path.join(save_root, subset)
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, filename), mask_vol)


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


def get_luna16_subset_mapping(root):
    paths = get_files(root, 'mhd')
    subset_mapping = {}
    for path in paths:
        folder, filename = os.path.split(path)
        pid = filename[:-4]
        subset  = os.path.split(folder)[1]
        subset_mapping[pid] = subset
    return subset_mapping


def extract_luna16_mask_from_pylidc(luna16_root, save_path, num_radiologist=3, padding=512, mask_threshold=8):
    df = pd.read_csv(os.path.join(luna16_root, 'annotations.csv'))
    pid2subset = get_luna16_subset_mapping(luna16_root)

    LIDC_IDRI_list= [f for f in os.listdir(DICOM_DIR) if f.startswith('LIDC-IDRI') and os.path.isdir(os.path.join(DICOM_DIR, f))]
    LIDC_IDRI_list.sort()
    # LIDC_IDRI_list = LIDC_IDRI_list[64:]
    # LIDC_IDRI_list = ['LIDC-IDRI-0332', 'LIDC-IDRI-0340', 'LIDC-IDRI-0388', 'LIDC-IDRI-0404']

    num = 0
    scan_table = {}
    freq = df['seriesuid'].value_counts()
    nn, ss = [], []
    num_1, num_2, num_3, num_4 = 0, 0, 0, 0,
    total_diff = []
    for lidc_id in tqdm(LIDC_IDRI_list):
        scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == lidc_id)
        num_scan_in_one_patient = scans.count()
        scan_list = scans.all()
        # scan_list[0:3]
        
        num_for_scan = 0
        for scan_idx, scan in enumerate(scan_list):
            if scan.series_instance_uid in df['seriesuid'].values:
                pid = scan.series_instance_uid
                subset = pid2subset[pid]
                short_pid = pid.split('.')[-1]
                nodules_annotation = scan.cluster_annotations()
                vol = scan.to_volume()
                out_mask = np.zeros_like(vol, dtype=np.uint8)
                # print(vol.shape)
                
                total_cmask = []
                for nodule_idx, nodule in enumerate(nodules_annotation):
                    # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
                    # This current for loop iterates over total number of nodules in a single patient
                    

                    # if len(nodule) >= 3:
                    cmask, cbbox, masks = radiologist_consensus(nodule, num_radiologist, 0)
                    # print(np.sum(cmask))

                    if np.sum(cmask):
                        # We calculate the malignancy information
                        malignancy, cancer_label = calculate_malignancy(nodule)

                        # save in semantic label
                        if  malignancy >= 3:
                            classes = 2
                        else:
                            classes = 1

                        out_mask[cbbox] = np.uint8(cmask) * classes
                        # for nodule_slice in range(cmask.shape[2]):
                        #     # This second for loop iterates over each single nodule.
                        #     # There are some mask sizes that are too small. These may hinder training.
                        #     if np.sum(cmask[:,:,nodule_slice]) <= mask_threshold:
                        #         continue
                            
                        #     # save in semantic label
                        #     if  malignancy >= 3:
                        #         classes = 2
                        #     else:
                        #         classes = 1

                        #     cmask[:,:,nodule_slice] = cmask[:,:,nodule_slice]*classes
                        num += 1
                        num_for_scan += 1

                #         total_cmask.append(cmask)
                # total_cmask = sum(total_cmask)
                total_cmask = out_mask
                total_cmask = np.transpose(total_cmask, (2, 0, 1))
                total_cmask = np.uint8(total_cmask)

                subset_dir = os.path.join(save_path, subset)
                os.makedirs(subset_dir, exist_ok=True)
                np.save(os.path.join(subset_dir, f'{short_pid}.npy'), total_cmask)
                        
                scan_table[pid] = {'num_luna': freq[pid], 'num_lidc': num_for_scan}
                if freq[pid] != num_for_scan:
                    diff_str = f'lidc_id: {lidc_id} {pid} luna: {freq[pid]} lidc: {num_for_scan}'
                    total_diff.append(diff_str)
                    print(diff_str)
                    ss.append(scan)
                    nn.append(nodules_annotation)
    # print(num_1, num_2, num_3, num_4)
    for s in total_diff:
        print(s)
    print(num)

    

def main():
    # input_root = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data'
    # save_root = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\luna16_hu_mask'
    # save_luna16_nodule_npy(input_root, save_root)

    # input_root = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data'
    # save_root = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\luna16_npy_raw'
    # save_luna16_nodule_npy(input_root, save_root)

    luna16_root = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data'
    save_path = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\luna16_mask'
    extract_luna16_mask_from_pylidc(luna16_root, save_path)


if __name__ == '__main__':
    main()
    # lidc_consensus()

    # raw = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data\subset0\1.3.6.1.4.1.14519.5.2.1.6279.6001.128023902651233986592378348912.mhd'
    # mask = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\luna16_mask\subset0\128023902651233986592378348912-0000.npy'
    # x, _, _, _ = load_itk(raw)
    # y = np.load(mask)
    # for i in range(x.shape[0]):
    #     if np.sum(y[...,i]):
    #         plt.imshow(x[i], 'gray')
    #         plt.imshow(y[...,i], alpha=0.1)
    #         plt.show()