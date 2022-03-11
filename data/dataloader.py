
from torch.utils.data import Dataset, DataLoader
import json
import cv2

from data.volume_generator import luna16_volume_generator, asus_nodule_volume_generator

def build_nodule_dataloader():
    pass


class TrainAsusNoduleDatset(Dataset):
    def __init__(self, coco_path):
        self.coco_path = coco_path
        with open(self.coco_path, newline='') as jsonfile:
            self.coco_data = json.load(jsonfile)
        print(3)

    def __len__(self):
        return len(self.coco_data['images'])
    
    def __getitem__(self, idx):
        image_infos = self.coco_data['images']
        annotations_infos = self.coco_data['images']

        file_name = image_infos[idx]['file_name']
        image = cv2.imread(file_name)
        # TODO: rle2mask
        segmentation =  annotations_infos[idx]['segmentation']
        
        # TODO: tansformation
        # if self.isDataAugmentation:
        #     image, segmentation = self.data_transformer(image, segmentation)
        sample = {'input': image, 'target': segmentation}
        return sample


class TestAsusNoduleDatset(Dataset):
    def __init__(self, coco_path, raw_path):
        # TODO: check case in coco
        self.coco_path = coco_path
        with open(self.coco_path, newline='') as jsonfile:
            self.coco_data = json.load(jsonfile)
        print(3)

    def __len__(self):
        return len(self.coco_data['images'])
    
    def __getitem__(self, idx):


if __name__ == '__main__':
    coco_path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodule\ASUS-Benign\coco\Nodule_Detection\cv-5\0\annotations_test.json'
    coco_dataset = TrainAsusNoduleDatset(coco_path)