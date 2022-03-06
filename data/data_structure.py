
import numpy as np


class StudyofLungNodule():
    def __init__(self, study_id, category_volume, raw_volume=None):
        self.study_id = study_id
        self.category_volume = category_volume
        self.nodule_ids = np.unique(category_volume).tolist()
        self.nodule_ids.remove(0)
        self.nodule_num = len(self.nodule_ids)
        self.raw_volume = raw_volume
        self.nodule_instances = self.build_nodule_instance()

    def build_nodule_instance(self):
        nodule_mapping = {}
        for id in self.nodule_ids:
            nodule_volume = np.where(self.category_volume==id, 1, 0)
            if self.raw_volume is not None:
                hu = np.mean(nodule_volume*self.raw_volume[...,0])
            else:
                hu = None
            nodule_mapping[id] = Nodule(self.study_id, id, nodule_volume, hu)
        return nodule_mapping

    def get_binary_volume(self):
        return np.where(self.category_volume>0, 1, 0)
    

class Nodule():
    def __init__(self, study_id, id, nodule_volume, hu=None):
        # TODO: Can self.nodule_volume cause memory problem if there are so many?
        self.study_id = study_id
        self.id = id
        self.nodule_volume = nodule_volume
        self.hu = hu

        self.nodule_size = np.sum(self.nodule_volume)
        self.nodule_exist_pixels = np.where(self.nodule_volume)
        self.nodule_position = self.get_nodule_position()

    def get_nodule_position(self):
        zs, ys, xs = self.nodule_exist_pixels
        nodule_range = {
            'index': {'min': np.min(zs), 'max': np.max(zs), 'center': np.mean(zs)},
            'row': {'min': np.min(ys), 'max': np.max(ys), 'center': np.mean(ys)},
            'column': {'min': np.min(xs), 'max': np.max(xs), 'center': np.mean(xs)},
        }
        return nodule_range
