import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import json, itertools
import os
import cv2
from volume_generator import lidc_to_datacatlog_valid, lidc_volume_generator, luna16_volume_generator, asus_nodule_volume_generator
from utils import raw_preprocess, mask_preproccess


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
    cats =[{'name':name, 'id':id} for name,id in cat_ids.items()]
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


def volume_to_coco_structure(cat_ids, volume_generator, seg_root=None):
    cat_ids = {'nodule': 1}    
    cats = [{'name':name, 'id':id} for name,id in cat_ids.items()]
    images, annotations = [], []
    idx = 0
    
    for vol_idx, (vol, mask_vol, infos) in enumerate(volume_generator):
        pid, scan_idx = infos['pid'], infos['scan_idx']
        pred_vol = np.zeros_like(mask_vol)
        for img_idx in range(vol.shape[2]):
            if img_idx%10 == 0:
                print(f'Patient {pid} Scan {scan_idx} slice {img_idx}')
            img = vol[...,img_idx]
            # img = raw_preprocess(img, lung_segment=False, norm=False)
            images.append(img)
            
            mask = mask_vol[...,img_idx]
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


def lidc_to_datacatlog_valid():
    # mode = 'valid'
    # CSV_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-Preprocessing\Metameta_info.csv'
    # df = pd.read_csv(CSV_PATH)
    # split = 16219
    # all_ids = df.original_image.unique()
    # if mode == 'train':
    #     samples = df[df.original_image.isin(all_ids[:split])]
    # elif mode == 'valid':
    #     samples = df[df.original_image.isin(all_ids[split:])]
    # data_root = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-Preprocessing-png\Image'
    # seg_root = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-Preprocessing-png\Mask'
    # with open('annotations_valid.json', 'w', encoding='utf-8') as jsonfile:
    #     json.load(valid_root, jsonfile, ensure_ascii=True, indent=4)

    with open('annotations_valid.json', newline='') as jsonfile:
        data_dict = json.load(jsonfile)
    return data_dict['images']


def xx():
    vol_generator = lidc_volume_generator(data_root, case_indices=case_indices, only_nodule_slices=only_nodules)

def main():
    DATA_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-Preprocessing-png\Image'
    GT_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-Preprocessing-png\Mask'
    CSV_PATH = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-Preprocessing\Metameta_info.csv'

    ## run it on first three images for demonstration:
    df = pd.read_csv(CSV_PATH)
    split = 16219
    all_ids = df.original_image.unique()
    train_sample = df[df.original_image.isin(all_ids[:split])]
    valid_sample = df[df.original_image.isin(all_ids[split:])]

    train_root = lidc_to_coco_structure(train_sample, data_root=DATA_PATH, seg_root=GT_PATH)
    valid_root = lidc_to_coco_structure(valid_sample, data_root=DATA_PATH, seg_root=GT_PATH)

    with open('annotations_train.json', 'w', encoding='utf-8') as jsonfile:
        json.dump(train_root, jsonfile, ensure_ascii=True, indent=4)

    with open('annotations_valid.json', 'w', encoding='utf-8') as jsonfile:
        json.dump(valid_root, jsonfile, ensure_ascii=True, indent=4)

if __name__ == '__main__':
    main()