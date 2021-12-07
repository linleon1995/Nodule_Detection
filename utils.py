import os
import cv2
import numpy as np
import site_path

from modules.data import dataset_utils


def convert_npy_to_png(src, dst, src_format, dst_format):
    src_files = dataset_utils.get_files(src, src_format)
    if not os.path.isdir(dst):
        os.makedirs(dst)
    for idx, f in enumerate(src_files):
        print(idx, f)
        img = np.load(f)
        if img.dtype == bool:
            img = np.uint8(img)
        else:
            img = np.uint8(255*((img-np.min(img))/(np.max(img)-np.min(img))))
        new_f = os.path.join(os.path.split(f)[0].replace(src, dst), os.path.split(f)[1].replace(src_format, dst_format))
        if not os.path.isdir(os.path.split(new_f)[0]):
            os.makedirs(os.path.split(new_f)[0])
        cv2.imwrite(new_f, img)


if __name__ == '__main__':
    src = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-Preprocessing\Mask'
    dst = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-Preprocessing-png\Mask'
    src_format = 'npy'
    dst_format = 'png'
    
    convert_npy_to_png(src, dst, src_format, dst_format)