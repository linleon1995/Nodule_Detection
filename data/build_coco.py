# import sys
# from pprint import pprint
# pprint(sys.path)

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from tqdm.contrib import tzip
import json, itertools
import os
import cv2

from data_utils import cv2_imshow, split_individual_mask, merge_near_masks
import site_path
from modules.data import dataset_utils



class coco_structure_converter():
    def __init__(self, cat_ids):
        self.images = []
        self.annotations = []
        self.cat_ids = cat_ids
        self.cats =[{'name':name, 'id':id} for name, id in self.cat_ids.items()]
        self.idx = 0

    def sample(self, img_path, mask, image_id, category):
        if np.sum(mask):
            image = {'id': image_id, 'width':512, 'height':512, 'file_name': f'{img_path}'}
            self.images.append(image)

            ys, xs = np.where(mask)
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            enc =binary_mask_to_rle(mask)
            seg = {
                'segmentation': enc, 
                'bbox': [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)],
                'area': int(np.sum(mask)),
                'image_id': image_id, 
                'category_id': self.cat_ids[category], 
                'iscrowd': 0, 
                'id':self.idx
            }
            self.idx += 1
            self.annotations.append(seg)


    def create_coco_structure(self):
        return {'categories':self.cats, 'images': self.images, 'annotations': self.annotations}


# From https://www.kaggle.com/stainsby/fast-tested-rle
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


# From https://newbedev.com/encode-numpy-array-using-uncompressed-rle-for-coco-dataset
def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    x = binary_mask.ravel(order='F')
    y = itertools.groupby(x)
    z = enumerate(y)
    for i, (value, elements) in enumerate(itertools.groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def coco_structure(train_df):
    cat_ids = {name:id+1 for id, name in enumerate(train_df.cell_type.unique())}    
    cats =[{'name':name, 'id':id} for name, id in cat_ids.items()]
    images = [{'id':id, 'width':row.width, 'height':row.height, 'file_name': f'train/{id}.png'} for id, row in train_df.groupby('id').agg('first').iterrows()]
    annotations=[]
    for idx, row in tqdm(train_df.iterrows()):
        mask = rle_decode(row.annotation, (row.height, row.width))
        ys, xs = np.where(mask)
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        enc =binary_mask_to_rle(mask)
        seg = {
            'segmentation':enc, 
            'bbox': [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)],
            'area': int(np.sum(mask)),
            'image_id':row.id, 
            'category_id':cat_ids[row.cell_type], 
            'iscrowd':0, 
            'id':idx
        }
        annotations.append(seg)
    return {'categories':cats, 'images':images,'annotations':annotations}


def build_coco_structure_split(input_paths_split, target_paths_split, cat_ids, area_threshold):
    coco_data_split = {}
    for data_split in input_paths_split:
        assert data_split in ('train', 'valid', 'test')
        input_paths, target_paths = input_paths_split[data_split], target_paths_split[data_split]
        coco_data = build_coco_structure(input_paths, target_paths, cat_ids, area_threshold)
        coco_data_split[data_split] = coco_data
    return coco_data_split


def build_coco_structure(input_paths, target_paths, cat_ids, area_threshold, category):
    coco_converter = coco_structure_converter(cat_ids)

    for img_path, mask_path in zip(input_paths, target_paths):
        mask = cv2.imread(mask_path)
        image_id = os.path.split(img_path)[1][:-4]
        splited_mask = split_individual_mask(mask[...,0])
        splited_mask = merge_near_masks(splited_mask)
        if len(splited_mask):
            for i, mask in enumerate(splited_mask, 1):
                if np.sum(mask) > area_threshold:
                    coco_converter.sample(img_path, mask, image_id, category)

    return coco_converter.create_coco_structure()


def merge_coco_structure(coco_group):
    output_coco = coco_group[0]
    for idx in range(1, len(coco_group)):
        # TODO: correct categories (if two different cat combine)
        # output_coco['categories'].update(coco_group[idx]['categories'])
        output_coco['images'].extend(coco_group[idx]['images'])
        output_coco['annotations'].extend(coco_group[idx]['annotations'])
    return output_coco
        
    
def build_asus_nodule_coco(convert_data_infos):
    coco_structures, all_target_paths = {}, {}
    for data_info in convert_data_infos:
        data_name = data_info.data_name
        data_path = data_info.data_path
        save_path = data_info.save_path
        split_indices = data_info.split_indices
        cat_ids = data_info.cat_ids
        area_threshold = data_info.area_threshold
        
        subset_image_path = os.path.join(data_path, 'Image')
        subset_mask_path = os.path.join(data_path, 'Mask')
        case_images = dataset_utils.get_files(subset_image_path, recursive=False, get_dirs=True)
        case_masks = dataset_utils.get_files(subset_mask_path, recursive=False, get_dirs=True)
        
        if not os.path.isdir(os.path.join(save_path, data_name)):
            os.makedirs(os.path.join(save_path, data_name))
            
        for split_name, indices in split_indices.items():
            split_images, split_masks = np.take(case_images, indices).tolist(), np.take(case_masks, indices).tolist()

            image_paths, target_paths = [], []
            for split_image, split_mask in zip(split_images, split_masks):
                image_paths.extend(dataset_utils.get_files(split_image, 'png', recursive=False))
                target_paths.extend(dataset_utils.get_files(split_mask, 'png', recursive=False))
                # assert len(image_paths) == len(target_paths), f'Inconsitent slice number Raw {len(image_paths)} Mask {len(target_paths)}'
           
            coco_structure = build_coco_structure(
                image_paths, target_paths, cat_ids, area_threshold, category=data_name)
            
            if split_name in coco_structures:
                coco_structures[split_name].append(coco_structure)
            else:
                coco_structures[split_name] = [coco_structure]
                
    for split_name in coco_structures:
        merge_coco =   merge_coco_structure(coco_structures[split_name])
        with open(os.path.join(save_path, f'annotations_{split_name}.json'), 'w', encoding='utf-8') as jsonfile:
            json.dump(merge_coco, jsonfile, ensure_ascii=True, indent=4)
                

def asus_nodule_to_coco_structure(data_path, split_rate=[0.7, 0.1, 0.2], area_threshold=30):
    subset_image_path = os.path.join(data_path, 'Image')
    subset_mask_path = os.path.join(data_path, 'Mask')
    subset_image_list = dataset_utils.get_files(subset_image_path, recursive=False, get_dirs=True)
    subset_mask_list = dataset_utils.get_files(subset_mask_path, recursive=False, get_dirs=True)

    # subset_image_list = subset_image_list[::-1]
    # subset_mask_list = subset_mask_list[::-1]

    cat_ids = cat_ids = {'nodule': 1}
    train_converter = coco_structure_converter(cat_ids)
    valid_converter = coco_structure_converter(cat_ids)
    test_converter = coco_structure_converter(cat_ids)
    # def decide_subset(num_sample, split_rate):
    #     # TODO: why sum(split_rate) = 0.999...
    #     # assert sum(split_rate) == 1.0, 'Split rate error'
    #     case_split_indices = (int(num_sample*split_rate[0]), int(num_sample*(split_rate[0]+split_rate[1])))
    #     return case_split_indices
    # case_split_indices = decide_subset(num_sample=len(subset_image_list), split_rate=split_rate)
    
    case_split_indices = (17, 19) # benign
    case_split_indices = (34, 36) # malignant
    

    for case_idx, (case_img_dir, case_mask_dir) in enumerate(tzip(subset_image_list, subset_mask_list)):
        case_image_list = dataset_utils.get_files(case_img_dir, 'png', recursive=False)
        case_mask_list = dataset_utils.get_files(case_mask_dir, 'png', recursive=False)
        assert len(case_image_list) == len(case_mask_list), f'Inconsitent slice number Raw {len(case_image_list)} Mask {len(case_mask_list)}'
        for img_path, mask_path in zip(case_image_list, case_mask_list):
            mask = cv2.imread(mask_path)
            image_id = os.path.split(img_path)[1][:-4]
            splited_mask = split_individual_mask(mask[...,0])
            splited_mask = merge_near_masks(splited_mask)

            if len(splited_mask):
                for i, mask in enumerate(splited_mask, 1):
                    if np.sum(mask) > area_threshold:
                        # cv2_imshow(255*np.tile(mask[...,np.newaxis], (1,1,3)), os.path.join('plot', 'ASUS_nodule', f'{image_id}-s{i}.png'))

                        if case_idx < case_split_indices[0]:
                            train_converter.sample(img_path, mask, image_id)
                        elif case_idx >= case_split_indices[0] and case_idx < case_split_indices[1]:
                            valid_converter.sample(img_path, mask, image_id)
                        else:
                            test_converter.sample(img_path, mask, image_id)
                        
    return train_converter.create_coco_structure(), valid_converter.create_coco_structure(), test_converter.create_coco_structure()


def luna16_to_coco_structure(data_path, label_type, split_rate=0.7, area_threshold=30):
    subset_list = dataset_utils.get_files(data_path, recursive=False, get_dirs=True)
    cat_ids = cat_ids = {'nodule': 1}
    train_converter = coco_structure_converter(cat_ids)
    valid_converter = coco_structure_converter(cat_ids)
    test_converter = coco_structure_converter(cat_ids)
    subset_list = subset_list[:8]
    for subset_idx, subset_path in enumerate(subset_list):
        subset = os.path.split(subset_path)[1]
        subset_image_dir = os.path.join(subset_path, 'Image')
        subset_mask_dir = os.path.join(subset_path, 'Mask')
        subset_image_list = dataset_utils.get_files(subset_image_dir, recursive=False, get_dirs=True)
        subset_mask_list = dataset_utils.get_files(subset_mask_dir, recursive=False, get_dirs=True)
        assert len(subset_image_list) == len(subset_mask_list), f'Inconsitent patient number Raw {len(subset_image_list)} Mask {len(subset_mask_list)}'
        state = 'train' if subset_idx <= 6 else 'valid'
        print(f'{state}-{subset}')
        for case_img_dir, case_mask_dir in tzip(subset_image_list, subset_mask_list):
            case_image_list = dataset_utils.get_files(case_img_dir, 'png', recursive=False)
            case_mask_list = dataset_utils.get_files(case_mask_dir, 'png', recursive=False)
            assert len(case_image_list) == len(case_mask_list), f'Inconsitent slice number Raw {len(case_image_list)} Mask {len(case_mask_list)}'
            for img_path, mask_path in zip(case_image_list, case_mask_list):
                mask = cv2.imread(mask_path)
                image_id = os.path.split(img_path)[1][:-4]

                if label_type == 'DLP':
                    mask_group = split_individual_mask(mask[...,0])
                    mask_group = merge_near_masks(mask_group)
                    # TO check every split masks
                    if len(mask_group) > 0:
                        for i, split_mask in enumerate(mask_group, 1):
                            if np.sum(split_mask) > area_threshold:
                                cv2_imshow(255*np.tile(split_mask[...,np.newaxis], (1,1,3)), os.path.join('plot', f'{image_id}-s{i}.png'))
                elif label_type == 'round':
                    mask_group = [mask[...,0]]
                else:
                    raise ValueError('Unknown')
            
                for mask in mask_group:
                    if np.sum(mask) > area_threshold:
                        if subset_idx <= 6:
                            train_converter.sample(img_path, mask, image_id)
                        elif subset_idx == 7:
                            valid_converter.sample(img_path, mask, image_id)
                        else:
                            test_converter.sample(img_path, mask, image_id)

    return train_converter.create_coco_structure(), valid_converter.create_coco_structure(), test_converter.create_coco_structure()


def luna16_round_to_coco_main():
    DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess-round\raw'
    annotation_root = os.path.join('Annotations', 'LUNA16-round')
    if not os.path.isdir(annotation_root):
        os.makedirs(annotation_root)

    train_root, valid_root, test_root = luna16_to_coco_structure(DATA_PATH, label_type='round')

    with open(os.path.join(annotation_root, 'annotations_train.json'), 'w', encoding='utf-8') as jsonfile:
        json.dump(train_root, jsonfile, ensure_ascii=True, indent=4)

    with open(os.path.join(annotation_root, 'annotations_valid.json'), 'w', encoding='utf-8') as jsonfile:
        json.dump(valid_root, jsonfile, ensure_ascii=True, indent=4)


def luna16_to_coco_main():
    DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\raw'
    annotation_root = os.path.join('Annotations', 'LUNA16')
    if not os.path.isdir(annotation_root):
        os.makedirs(annotation_root)

    train_root, valid_root, test_root = luna16_to_coco_structure(DATA_PATH, label_type='DLP')

    with open(os.path.join(annotation_root, 'annotations_train.json'), 'w', encoding='utf-8') as jsonfile:
        json.dump(train_root, jsonfile, ensure_ascii=True, indent=4)

    with open(os.path.join(annotation_root, 'annotations_valid.json'), 'w', encoding='utf-8') as jsonfile:
        json.dump(valid_root, jsonfile, ensure_ascii=True, indent=4)


def asus_benign_to_coco_main():
    DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\benign\raw_merge'
    annotation_root = os.path.join('Annotations', 'ASUS_Nodule', 'benign_merge')
    if not os.path.isdir(annotation_root):
        os.makedirs(annotation_root)

    train_root, valid_root, test_root = asus_nodule_to_coco_structure(DATA_PATH, area_threshold=8)

    with open(os.path.join(annotation_root, 'annotations_train.json'), 'w', encoding='utf-8') as jsonfile:
        json.dump(train_root, jsonfile, ensure_ascii=True, indent=4)

    with open(os.path.join(annotation_root, 'annotations_valid.json'), 'w', encoding='utf-8') as jsonfile:
        json.dump(valid_root, jsonfile, ensure_ascii=True, indent=4)

    with open(os.path.join(annotation_root, 'annotations_test.json'), 'w', encoding='utf-8') as jsonfile:
        json.dump(test_root, jsonfile, ensure_ascii=True, indent=4)


def asus_malignant_to_coco_main():
    DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\malignant\raw_merge'
    annotation_root = os.path.join('Annotations', 'ASUS_Nodule', 'malignant_merge')
    if not os.path.isdir(annotation_root):
        os.makedirs(annotation_root)

    train_root, valid_root, test_root = asus_nodule_to_coco_structure(DATA_PATH, area_threshold=8)

    with open(os.path.join(annotation_root, 'annotations_train.json'), 'w', encoding='utf-8') as jsonfile:
        json.dump(train_root, jsonfile, ensure_ascii=True, indent=4)

    with open(os.path.join(annotation_root, 'annotations_valid.json'), 'w', encoding='utf-8') as jsonfile:
        json.dump(valid_root, jsonfile, ensure_ascii=True, indent=4)

    with open(os.path.join(annotation_root, 'annotations_test.json'), 'w', encoding='utf-8') as jsonfile:
        json.dump(test_root, jsonfile, ensure_ascii=True, indent=4)


# TODO: use dict or nametuple?
class DatasetConvertInfo():
    def __init__(self, data_name, data_path, split_indices, area_threshold, cat_ids, save_path):
        self.data_name = data_name
        self.data_path = data_path
        self.split_indices = split_indices
        self.area_threshold = area_threshold
        self.cat_ids = cat_ids
        self.save_path = save_path
        
        
def build_asus_malignant():
    # General
    # TODO: make from config?
    save_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\Annotations\test'
    # cat_ids = {'nodule': 1}
    cat_ids = {'benign': 1, 'malignant': 2}
    area_threshold = 8
    
    # Benign
    name = 'benign'
    benign_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\benign\raw_merge'
    train_indices = list(range(0, 17))
    valid_indices = list(range(17, 19))
    test_indices = list(range(19, 25))
    split_indices = {'train': train_indices,
                     'valid': valid_indices,
                     'test': test_indices}
    benign_convert_info = DatasetConvertInfo(
        name, benign_path, split_indices, area_threshold, cat_ids, save_path)
    
    # Malignant
    name = 'malignant'
    malignant_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\malignant\raw_merge'
    train_indices = list(range(0, 34))
    valid_indices = list(range(34, 36))
    test_indices = list(range(36, 44))
    split_indices = {'train': train_indices,
                     'valid': valid_indices,
                     'test': test_indices}
    malignant_convert_info = DatasetConvertInfo(
        name, malignant_path, split_indices, area_threshold, cat_ids, save_path)
    
    convert_infos = [benign_convert_info, malignant_convert_info]
    build_asus_nodule_coco(convert_infos)


def main():
    # luna16_round_to_coco_main()
    # luna16_to_coco_main()
    # asus_benign_to_coco_main()
    # asus_malignant_to_coco_main()
    build_asus_malignant()


if __name__ == '__main__':
    main()
    pass