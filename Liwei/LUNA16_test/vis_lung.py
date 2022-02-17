'''
Modified Date: 2021/12/14
Author: Li-Wei Hsiao
mail: nfsmw308@gmail.com
'''

import SimpleITK as sitk
import numpy as np
import matplotlib.pylab as plt
import cv2, os, tqdm

def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing

if __name__ in "__main__":
    filename = '../dataset/data-unversioned/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd'
    filename_mask = '../dataset/data-unversioned/seg-lungs-LUNA16/1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd'
    ct_scans, origin, spacing = load_itk(filename)
    ct_scans_mask, origin_mask, spacing_mask = load_itk(filename_mask)
    save_path = 'gt_lung'
    os.makedirs(save_path, exist_ok=True)
    for idx, (ct_t, pos_t) in enumerate(tqdm.tqdm(zip(ct_scans, ct_scans_mask))):
        plt.imshow(ct_t, cmap='gray')
        # pos_t[pos_t != 5] = 0
        # pos_t[pos_t == 5] = 1
        if pos_t.sum() != 0:
            # print(np.unique(pos_t))
            plt.contour(pos_t, 5, cmap='Reds')
        plt.axis('off')
        plt.savefig(f"{save_path}/{idx}.png")
        # plt.show()
        plt.close('all')
        