import numpy as np
import os

from data.data_utils import get_files
from data.volume_generator import luna16_volume_generator


def save_luna16_nodule_mask_npy(input_root, save_root):
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


def main():
    input_root = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data'
    save_root = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\luna16_hu_mask'
    save_luna16_nodule_mask_npy(input_root, save_root)


if __name__ == '__main__':
    main()