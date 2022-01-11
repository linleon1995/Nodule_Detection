import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from tqdm.contrib import tzip
import json, itertools
import os
import cv2
from volume_generator import lidc_volume_generator, luna16_volume_generator, asus_nodule_volume_generator
from utils import cv2_imshow, raw_preprocess, mask_preprocess, split_individual_mask, merge_near_masks

import site_path
from modules.data import dataset_utils



class coco_structure_converter():
    def __init__(self, cat_ids):
        self.images = []
        self.annotations = []
        self.cat_ids = cat_ids
        self.cats =[{'name':name, 'id':id} for name, id in self.cat_ids.items()]
        self.idx = 0

    def sample(self, img_path, mask, image_id):
        # TODO: Temporally keep image with object because no idea to deal with the label of objectless case
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
                'category_id': self.cat_ids['nodule'], 
                'iscrowd': 0, 
                'id':self.idx
            }
            self.idx += 1
            self.annotations.append(seg)


    def create_coco_structure(self):
        return {'categories':self.cats, 'images': self.images,'annotations': self.annotations}


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


def asus_nodule_to_coco_structure(data_path, split_rate=[0.7, 0.1, 0.2], area_threshold=30):
    subset_image_path = os.path.join(data_path, 'Image')
    subset_mask_path = os.path.join(data_path, 'Mask')
    subset_image_list = dataset_utils.get_files(subset_image_path, recursive=False, get_dirs=True)
    subset_mask_list = dataset_utils.get_files(subset_mask_path, recursive=False, get_dirs=True)
    cat_ids = cat_ids = {'nodule': 1}
    train_converter = coco_structure_converter(cat_ids)
    valid_converter = coco_structure_converter(cat_ids)
    test_converter = coco_structure_converter(cat_ids)
    def decide_subset(num_sample, split_rate):
        # TODO: why sum(split_rate) = 0.999...
        # assert sum(split_rate) == 1.0, 'Split rate error'
        case_split_indices = (int(num_sample*split_rate[0]), int(num_sample*(split_rate[0]+split_rate[1])))
        return case_split_indices
    case_split_indices = decide_subset(num_sample=len(subset_image_list), split_rate=split_rate)
    
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
                        cv2_imshow(255*np.tile(mask[...,np.newaxis], (1,1,3)), os.path.join('plot', 'ASUS_nodule', f'{image_id}-s{i}.png'))

                        if case_idx < case_split_indices[0]:
                            train_converter.sample(img_path, mask, image_id)
                        elif case_idx >= case_split_indices[0] and case_idx < case_split_indices[1]:
                            valid_converter.sample(img_path, mask, image_id)
                        else:
                            test_converter.sample(img_path, mask, image_id)
                        
    return train_converter.create_coco_structure(), valid_converter.create_coco_structure(), test_converter.create_coco_structure()


def luna16_to_coco_structure(data_path, split_rate=0.7, area_threshold=30):
    subset_list = dataset_utils.get_files(data_path, recursive=False, get_dirs=True)
    cat_ids = cat_ids = {'nodule': 1}
    train_converter = coco_structure_converter(cat_ids)
    valid_converter = coco_structure_converter(cat_ids)
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
                splited_mask = split_individual_mask(mask[...,0])
                splited_mask = merge_near_masks(splited_mask)

                if len(splited_mask) > 0:
                    # print(3)
                    for i, mm in enumerate(splited_mask, 1):
                        # print(np.sum(mm))
                        if np.sum(mm) > area_threshold:
                            cv2_imshow(255*np.tile(mm[...,np.newaxis], (1,1,3)), os.path.join('plot', f'{image_id}-s{i}.png'))

                for mask in splited_mask:
                    if np.sum(mask) > area_threshold:
                        if subset_idx <= 6:
                            train_converter.sample(img_path, mask, image_id)
                        elif subset_idx == 7:
                            valid_converter.sample(img_path, mask, image_id)
    return train_converter.create_coco_structure(), valid_converter.create_coco_structure()



def volume_to_coco_structure(cat_ids, volume_generator, seg_root=None):
    cats = [{'name':name, 'id':id} for name,id in cat_ids.items()]
    images, annotations = [], []
    idx = 0
    
    for vol_idx, (vol, mask_vol, infos) in enumerate(volume_generator):
        dataset, pid, subset, scan_idx = infos['dataset'], infos['subset'], infos['pid'], infos['scan_idx']
        # pred_vol = np.zeros_like(mask_vol)
        for img_idx in range(vol.shape[0]):
            if img_idx%10 == 0:
                print(f'Patient {pid} Scan {scan_idx} slice {img_idx}')
            img = vol[img_idx]
            # img = raw_preprocess(img, lung_segment=False, norm=False)
            images.append(img)
            
            mask = mask_vol[img_idx]

            # TODO: Temporally keep image with object because no idea to deal with the label of objectless case
            if not np.sum(mask):
                # print(np.sum(mask))
                continue

            
            # image_id = f'{dataset}-{subset}-Case{vol_idx}-{img_idx:03d}'
            # image = {'id': image_id, 'width':512, 'height':512, 'file_name': f'{os.path.join(data_root, _dir, file_name)}'}
        

            # mask = mask_preproccess(mask)
            ys, xs = np.where(mask)
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            enc =binary_mask_to_rle(mask)
            seg = {
                'segmentation':enc, 
                'bbox': [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)],
                'area': int(np.sum(mask)),
                'image_id': f'{pid}-Scan{scan_idx}-Slice{img_idx:04d}', 
                'category_id':cat_ids['nodule'], 
                'iscrowd':0, 
                'id':idx
            }
            annotations.append(seg)
            idx += 1
        return {'categories':cats, 'images':images,'annotations':annotations}
    
    
def lidc_to_coco_structure(df, data_root, seg_root=None):
    cat_ids = {'nodule': 1,
               'malignant': 2}    
    cats = [{'name': 'benign', 'id': 1}, 
            {'name': 'malignant', 'id': 2}]
    images = []

    annotations=[]
    for idx, row in tqdm(df.iterrows()):
        if 'CN' in row.original_image:
            continue
        
        # mask = rle_decode(row.annotation, (row.height, row.width))
        file_name = f'{row.original_image}.png'
        _dir = 'LIDC-IDRI-' + row.original_image.split('_')[0]
        image = {'id': row.original_image, 'width':512, 'height':512, 'file_name': f'{os.path.join(data_root, _dir, file_name)}'}
        if seg_root:
            seg_file_name = file_name.replace('NI', 'MA')
            image['sem_seg_file_name'] = f'{os.path.join(seg_root, _dir, seg_file_name)}'

        images.append(image)

        mask = cv2.imread(image['file_name'].replace('Image', 'Mask').replace('NI', 'MA'))
        mask = mask[...,0]
        # mask = np.load(image['file_name'].replace('Image', 'Mask').replace('NI', 'MA'))
        ys, xs = np.where(mask)
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        enc =binary_mask_to_rle(mask)
        category = 'malignant' if row.is_cancer=='True' or row.is_cancer=='Ambiguous'  else 'benign'
        seg = {
            'segmentation':enc, 
            'bbox': [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)],
            'area': int(np.sum(mask)),
            'image_id':row.original_image, 
            'category_id':cat_ids[category], 
            'iscrowd':0, 
            'id':idx
        }
        annotations.append(seg)
        print(idx, row.original_image, category)
    return {'categories':cats, 'images':images,'annotations':annotations}


def luun16_to_coco_main():
    DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\raw'
    annotation_root = os.path.join('Annotations', 'LUNA16')
    if not os.path.isdir(annotation_root):
        os.makedirs(annotation_root)

    train_root, valid_root = luna16_to_coco_structure(DATA_PATH)

    with open(os.path.join(annotation_root, 'annotations_train.json'), 'w', encoding='utf-8') as jsonfile:
        json.dump(train_root, jsonfile, ensure_ascii=True, indent=4)

    with open(os.path.join(annotation_root, 'annotations_valid.json'), 'w', encoding='utf-8') as jsonfile:
        json.dump(valid_root, jsonfile, ensure_ascii=True, indent=4)


def asus_nodule_to_coco_main():
    DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\malignant\raw'
    annotation_root = os.path.join('Annotations', 'ASUS_Nodule')
    if not os.path.isdir(annotation_root):
        os.makedirs(annotation_root)

    train_root, valid_root, test_root = asus_nodule_to_coco_structure(DATA_PATH, area_threshold=8)

    with open(os.path.join(annotation_root, 'annotations_train.json'), 'w', encoding='utf-8') as jsonfile:
        json.dump(train_root, jsonfile, ensure_ascii=True, indent=4)

    with open(os.path.join(annotation_root, 'annotations_valid.json'), 'w', encoding='utf-8') as jsonfile:
        json.dump(valid_root, jsonfile, ensure_ascii=True, indent=4)

    with open(os.path.join(annotation_root, 'annotations_test.json'), 'w', encoding='utf-8') as jsonfile:
        json.dump(test_root, jsonfile, ensure_ascii=True, indent=4)

if __name__ == '__main__':
    # luun16_to_coco_main()
    asus_nodule_to_coco_main()