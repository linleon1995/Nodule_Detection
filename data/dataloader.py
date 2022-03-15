
from torch.utils.data import Dataset, DataLoader
import json
import cv2
import numpy as np

from data.volume_generator import ASUSNoduleVolumeGenerator, luna16_volume_generator, asus_nodule_volume_generator
from data.build_coco import rle_decode


def build_nodule_dataloader():
    pass


def get_shift_index(cur_index, index_shift, boundary=None):
    start, end = cur_index-index_shift, cur_index+index_shift

    if boundary is not None:
        boundary_length = boundary[1] - boundary[0]
        sequence_length = 2*index_shift+1
        assert sequence_length < boundary_length
        assert boundary[1] > boundary[0]
        assert (cur_index>=0 and index_shift>=0 and boundary[0]>=0 and boundary[1]>=0)
        assert cur_index >= boundary[0]  or cur_index <= boundary[1]
        if start < boundary[0]:
            start, end = boundary[0], boundary[0]+sequence_length-1
        elif end > boundary[1]:
            start, end = boundary[1]-sequence_length, boundary[1]-1

    indices = np.arange(start, end+1)
    # print(cur_index, index_shift, boundary, indices)
    return indices


class SimpleNoduleDataset():
    def __init__(self, volume, slice_shift=3):
        self.volume = volume
        self.slice_shift = slice_shift

    def __len__(self):
        return self.volume.shape[0]

    def __getitem__(self, vol_idx):
        slice_indcies = get_shift_index(cur_index=vol_idx, index_shift=self.slice_shift, boundary=[0, self.volume.shape[0]-1])
        return self.volume[slice_indcies]


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