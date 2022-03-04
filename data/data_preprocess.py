import os

from data import medical_to_img
from data import build_coco
from data import asus_data_merge

import site_path
from modules.utils import configuration
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
from utils.volume_generator import luna16_volume_generator, asus_nodule_volume_generator

DATASET_NAME = ['ASUS-Benign', 'ASUS-Malignant'] # ASUS-Benign, ASUS-Malignant, LUNA16, LUNA16-Round
        
# TODO: Add some logging file or excel for recording   

def get_asus_data_split(dataset_name):
    if dataset_name == 'ASUS-Benign':
        train_split = list(range(17))
        valid_split = list(range(17, 19))
        test_split = list(range(19, 25))
    elif dataset_name == 'ASUS-Malignant':
        train_split = list(range(34))
        valid_split = list(range(34, 36))
        test_split = list(range(36, 44))
    else:
        raise ValueError('Unknown dataset name.')
    
    output_data_split = {}
    data_split = {'train': train_split, 'valid': valid_split, 'test': test_split}
    for split in data_split:
        for idx in data_split[split]:
            output_data_split[idx] = split

    return output_data_split
        

def build_parameters(dataset_name):
    if dataset_name not in ['ASUS-Benign', 'ASUS-Malignant']:
        return None

    cfg = configuration.load_config(f'data/config/{dataset_name}.yml', dict_as_member=True)
    data_root = cfg.PATH.DATA_ROOT
    merge_path = os.path.join(os.path.split(data_root)[0], 'merge')
    image_path = os.path.join(os.path.split(data_root)[0], 'image')
    coco_path = os.path.join(os.path.split(data_root)[0], 'coco')
    volume_generator = asus_nodule_volume_generator(data_path=merge_path)
    data_split = get_asus_data_split(dataset_name)

    data_parameters = {'data_root': data_root,
                       'merge_path': merge_path,
                       'image_path': image_path,
                       'coco_path': coco_path,
                       'volume_generator': volume_generator,
                       'data_split': data_split}
    return data_parameters
    

def data_preprocess(dataset_name):
    dataset_parameter = build_parameters(dataset_name)

    if dataset_parameter is not None:
        data_root = dataset_parameter['data_root']
        merge_path = dataset_parameter['merge_path']
        image_path = dataset_parameter['image_path']
        coco_path = dataset_parameter['coco_path']
        volume_generator = dataset_parameter['volume_generator']
        data_split = dataset_parameter['data_split']
        kc_image_path = image_path.replace('image', 'kc_image')

        for path in [merge_path, image_path, coco_path, kc_image_path]:
            if not os.path.isdir(path):
                os.makedirs(path)

        # Merge mhd data
        if dataset_name == 'ASUS-Benign':
            filekey = 'B'
        elif dataset_name == 'ASUS-Malignant':
            filekey = 'm'
        asus_data_merge.merge_asus_data(data_root, merge_path, filekey)

        # Convert medical 3d volume data to image format
        medical_to_img.volumetric_data_preprocess(save_path=image_path, volume_generator=volume_generator)
        medical_to_img.volumetric_data_preprocess_KC(data_split, save_path=kc_image_path, volume_generator=volume_generator)

        # # Build up coco-structure
        # build_coco.asus_nodule_to_coco_structure()


def main():
    for name in DATASET_NAME:
        data_preprocess(name)


if __name__ == '__main__':
    main()
    pass