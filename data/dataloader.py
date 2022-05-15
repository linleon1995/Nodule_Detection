
from torch.utils.data import Dataset, DataLoader
import json
import cv2
import numpy as np
import os

from data.volume_generator import TMHNoduleVolumeGenerator, luna16_volume_generator, asus_nodule_volume_generator
from data.data_utils import get_files, get_shift_index
from data.data_transformer import ImageDataTransformer
from data.volume_to_3d_crop import CropVolume
from data.crop_utils import CropVolumeOP
from dataset_conversion.build_coco import rle_decode

class GeneralDataset():
    def __init__(self, input_path_list, target_path_list, input_load_func, target_load_func, data_transformer=None):
        # TODO: random seed?
        # TODO: shape,  type check
        # TODO: data preprocess?
        # TODO: no label case
        # TODO: input and target in the same file
        # This is a general Pytorch dataset which should can be use in any place.
        # If the loading function and tansformer are implemented properly, the dataset should work.
        self.input_load_func = input_load_func
        self.target_load_func = target_load_func
        self.input_path_list = input_path_list
        self.target_path_list = target_path_list
        self.data_transformer = data_transformer
    
    def __len__(self):
        return len(self.input_path_list)

    def __getitem__(self, idx):
        input_data, target = self.input_load_func(self.input_path_list[idx]), self.target_load_func(self.target_path_list[idx])
        # if np.sum(target) > 0:
        # TODO: multi-class issue
        target = target[np.newaxis]
        target = np.concatenate([1-target, target], axis=0)

        # TODO: general implemtention for different dimesnion option (do this inside trnasformer)
        input_data = np.swapaxes(np.swapaxes(input_data, 0, 2), 0, 1)
        target = np.swapaxes(np.swapaxes(target, 0, 2), 0, 1)

        if self.data_transformer is not None:
            input_data, target, self.data_transformer(input_data, target)

        input_data = np.swapaxes(np.swapaxes(input_data, 0, 2), 1, 2)
        target = np.swapaxes(np.swapaxes(target, 0, 2), 1, 2)
        return {'input': input_data, 'target': target}



def build_dataloader(input_roots, target_roots, train_cases, valid_cases, train_batch_size, pin_memory=True, 
                     num_workers=0, transform_config=None):
    input_load_func = target_load_func = np.load
    def get_samples(roots, cases, load_format):
        data_dir = []
        for root in roots:
            data_dir.extend(get_files(root, keys=cases, get_dirs=True, recursive=False))
        samples = []
        for data_dir in data_dir:
            samples.extend(get_files(data_dir, keys=load_format))
        return samples

    train_input_samples = get_samples(input_roots, train_cases, 'npy')   
    valid_input_samples = get_samples(input_roots, valid_cases, 'npy')   
    train_target_samples = get_samples(target_roots, train_cases, 'npy')   
    valid_target_samples = get_samples(target_roots, valid_cases, 'npy')   
            
    # TODO: Temporally solution because of slowing validation
    # train_input_samples, train_target_samples = train_input_samples[:300], train_target_samples[:300]
    valid_input_samples, valid_target_samples = valid_input_samples[:500], valid_target_samples[:500]

    transformer = ImageDataTransformer(transform_config) if transform_config is not None else None
    train_dataset = GeneralDataset(
        train_input_samples, train_target_samples, input_load_func, target_load_func, data_transformer=transformer)
    valid_dataset = GeneralDataset(valid_input_samples, valid_target_samples, input_load_func, target_load_func)

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    return train_dataloader, valid_dataloader


# TODO: fix it
def change_eval_size(vol, mask_vol, size):
    vol = vol[...,0]
    vol = np.swapaxes(np.swapaxes(vol, 0, 2), 0, 1)
    vol = cv2.resize(vol, (size,size))
    vol = np.swapaxes(np.swapaxes(vol, 0, 2), 1, 2)
    vol = np.tile(vol[...,np.newaxis], (1, 1, 1, 3))

    # mask_vol = mask_vol[...,np.newaxis]
    mask_vol = cv2.resize(mask_vol, (size,size), interpolation=cv2.INTER_NEAREST)
    mask_vol = mask_vol[np.newaxis]
    # mask_vol = np.swapaxes(np.swapaxes(mask_vol, 0, 2), 1, 2)
    return vol, mask_vol

# TODO: for all volume converter, add raise error while volume doesn't send in
class SimpleNoduleDataset():
    def __init__(self, volume, slice_shift=3):
        self.volume = volume
        self.slice_shift = slice_shift

    def __len__(self):
        return self.volume.shape[0]

    def __getitem__(self, vol_idx):
        slice_indcies = get_shift_index(cur_index=vol_idx, index_shift=self.slice_shift, boundary=[0, self.volume.shape[0]-1])
        stack_images = self.volume[slice_indcies]
        return stack_images


class CropNoduleDataset():
    def __init__(self, volume, crop_range, crop_shift, convert_dtype=None, overlapping=1.0):
        self.cropping_op = CropVolume(crop_range, crop_shift, convert_dtype, overlapping)
        # self.cropping_op = CropVolumeOP(crop_range, crop_shift, convert_dtype, overlapping)
        self.crop_data = self.cropping_op(volume)
        for idx in range(len(self.crop_data)):
            self.crop_data[idx]['data'] = self.crop_data[idx]['data'][np.newaxis]

    def __len__(self):
        return len(self.crop_data)

    def __getitem__(self, sample_idx):
        return self.crop_data[sample_idx]


class NoduleDataset():
    def __init__(self, data_path, volume_generator, slice_shift=3):
        self.data_path = data_path
        self.slice_shift = slice_shift
        self.volume_generator = volume_generator
        self.first_trial = True
        self.index_converter = self.build_index_converter(self.volume_generator.case_pids)

    def __len__(self):
        return sum(list(self.volume_generator.total_num_slice.values()))

    def __getitem__(self, sample_idx):
        if self.first_trial:
            self.first_trial = False
            pid = self.index_converter[sample_idx]['pid']
            self.cur_pid = pid
            raw_vol, vol, mask_vol, origin, spacing, direction = self.volume_generator.get_data_by_pid_asus(self.data_path, pid)
            self.vol, self.mask_vol = vol, mask_vol

        pid = self.index_converter[sample_idx]['pid']
        if pid != self.cur_pid:
            raw_vol, vol, mask_vol, origin, spacing, direction = self.volume_generator.get_data_by_pid_asus(self.data_path, pid)
            self.vol, self.mask_vol = vol, mask_vol
            self.cur_pid = pid

        vol_idx = self.index_converter[sample_idx]['vol_idx']
        slice_indcies = get_shift_index(cur_index=vol_idx, index_shift=self.slice_shift, boundary=[0, self.vol.shape[0]-1])
        inputs = self.vol[slice_indcies]
        target = self.mask_vol[vol_idx]
        # print(slice_indcies, vol_idx, inputs.shape, target.shape)
        return {'input': inputs, 'target': target}

    def build_index_converter(self, case_pids):
        index_converter = []
        for pid in case_pids:
            raw_vol, vol, mask_vol, origin, spacing, direction = self.volume_generator.get_data_by_pid_asus(self.data_path, pid)
            for vol_idx in range(vol.shape[0]):
                index_converter.append({'pid': pid, 'vol_idx': vol_idx})
        return index_converter
        

class SimpleCocoDataset(Dataset):
    def __init__(self, coco_path, annot_type):
        self.coco_path = coco_path
        self.coco_data = self.build_coco(self.coco_path)
        self.categories = self.coco_data['categories']
        self.annot_type = annot_type
        # for  annot_type in self.annotation_type = self.coco_data['annotations']:
        #     if annot_type in ['segmentation', 'keypoint']
       
        # print(3)

    def __len__(self):
        return len(self.coco_data['images'])

    @classmethod
    def build_coco(cls, coco_path):
        with open(coco_path, newline='') as jsonfile:
            coco_data = json.load(jsonfile)
        
        # TODO: inspection, the annotation type should be determined automatically, and the coco_data should be check is fit the coco spec
        # TODO: Can this related to build_coco? (BaseCocoParser, BaseCocoBuilder, ...)
        return coco_data

class TrainAsusNoduleDatset(SimpleCocoDataset):
    def __init__(self, coco_path, annot_type):
        super().__init__(coco_path, annot_type)
       
        print(3)
    
    def __getitem__(self, idx):
        image_infos = self.coco_data['images']
        annotations_infos = self.coco_data['annotations']

        file_name = image_infos[idx]['file_name']
        image = cv2.imread(file_name)

        annotation =  annotations_infos[idx][self.annot_type]
        mask = rle_decode(annotation['counts'], annotation['size'])
        # TODO: tansformation
        # if self.isDataAugmentation:
        #     image, segmentation = self.data_transformer(image, segmentation)
        sample = {'input': image, 'target': mask}
        return sample


# class TestAsusNoduleDatset(Dataset):
#     def __init__(self, coco_path, raw_path):
#         # TODO: check case in coco
#         self.coco_path = coco_path
#         with open(self.coco_path, newline='') as jsonfile:
#             self.coco_data = json.load(jsonfile)
#         print(3)

#     def __len__(self):
#         return len(self.coco_data['images'])
    
#     def __getitem__(self, idx):


if __name__ == '__main__':
    # coco_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodule\ASUS-Benign\coco\Nodule_Detection\cv-5\0\annotations_test.json'
    # coco_dataset = TrainAsusNoduleDatset(coco_path, annot_type='segmentation')
    # coco_dataloader = DataLoader(coco_dataset, batch_size=2, shuffle=False, pin_memory=True, num_workers=0)
    # for sample in coco_dataloader:
    #     print(sample)

    
    # Come from generator
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodule\ASUS-Benign\merge'
    case_pids = ['1B0024', '1B0025']
    volume_generator = ASUSNoduleVolumeGenerator(data_path, case_pids)
    dataset = NoduleDataset(data_path, volume_generator)
    coco_dataloader = DataLoader(dataset, batch_size=2, shuffle=False, pin_memory=True, num_workers=0)
    for sample in coco_dataloader:
        # print(sample)
        print('test')