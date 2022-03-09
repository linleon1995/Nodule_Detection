
import numpy as np



# class EvalofLungNodule():
#     def __init__(self, study_id):
#         self.study_id = study_id
#         self.target_nodules = {}
#         self.pred_nodules = {}

#     def record(self, pred_nodule, pred_s):
#         self.pred_nodules[]

class LungNoduleStudy():
    def __init__(self, study_id, category_volume, raw_volume=None):
        self.study_id = study_id
        self.category_volume = category_volume
        self.raw_volume = raw_volume
        self.nodule_ids = np.unique(category_volume).tolist()
        self.nodule_ids.remove(0)
        self.nodule_num = len(self.nodule_ids)
        self.volume_shape = category_volume.shape
        self.study_evals = {}
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
    
    def add_score(self, score_name, score):
        self.study_evals[score_name] = score


class Nodule():
    def __init__(self, study_id, id, nodule_volume, hu=None):
        # TODO: Can self.nodule_volume cause memory problem if there are so many?
        self.study_id = study_id
        self.id = id
        self.nodule_volume = nodule_volume
        self.hu = hu
        self.nodule_score = {}

        self.nodule_size = np.sum(self.nodule_volume)
        self.nodule_exist_pixels = np.where(self.nodule_volume)
        self.nodule_range, self.nodule_center = self.get_nodule_position()

    def get_nodule_position(self):
        zs, ys, xs = self.nodule_exist_pixels
        nodule_range = {
            'index': {'min': np.min(zs), 'max': np.max(zs)},
            'row': {'min': np.min(ys), 'max': np.max(ys)},
            'column': {'min': np.min(xs), 'max': np.max(xs)},
        }
        nodule_center = {
            'index': np.mean(zs),
            'row': np.mean(ys),
            'column': np.mean(xs),
        }
        return nodule_range, nodule_center

    def add_score(self, score_name, score):
        self.nodule_score[score_name] = score