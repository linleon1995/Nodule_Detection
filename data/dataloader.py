
from torch.utils.data import Dataset, DataLoader
import json
import cv2

from data.volume_generator import luna16_volume_generator, asus_nodule_volume_generator
from data.build_coco import rle_decode


def build_nodule_dataloader():
    pass


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
    coco_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodule\ASUS-Benign\coco\Nodule_Detection\cv-5\0\annotations_test.json'
    coco_dataset = TrainAsusNoduleDatset(coco_path, annot_type='segmentation')
    coco_dataloader = DataLoader(coco_dataset, batch_size=2, shuffle=False, pin_memory=True, num_workers=0)
    for sample in coco_dataloader:
        print(sample)
