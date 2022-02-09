import numpy as np
import cc3d
CONNECTIVITY = 26
AREA_THRESHOLD = 20


class VolumePostprocessor():
    def __init__(self, connectivity=CONNECTIVITY, area_threshold=AREA_THRESHOLD, match_threshold=0.5):
        self.connectivity = connectivity
        self.area_threshold = area_threshold
        self.match_threshold = match_threshold

    @classmethod
    def postprocess(cls, binary_volume, connectivity=CONNECTIVITY, area_threshold=AREA_THRESHOLD):
        catrgories_volume = cc3d.connected_components(binary_volume, connectivity=connectivity)
        total_nodule_metadata = cls.build_nodule_metadata(catrgories_volume)
        volume_info = {'volume_array': catrgories_volume, 'total_nodule_metadata': total_nodule_metadata}
        if volume_info['total_nodule_metadata'] is not None:
            if area_threshold > 0:
                volume_info = cls.remove_small_area(volume_info, area_threshold)
            volume_info = cls.convert_label_value(volume_info)
        return volume_info

    @classmethod
    def remove_small_area(cls, volume_info, area_threshold):
        volume, total_nodule_metadata = volume_info['volume_array'], volume_info['total_nodule_metadata']
        keep_indices = list(range(len(total_nodule_metadata)))
        for idx, nodule_metadata in enumerate(total_nodule_metadata):
            if nodule_metadata['Nodule_size'] < area_threshold:
                keep_indices.remove(idx)
                volume[volume==nodule_metadata['Nodule_id']] = 0

        # Remove smaller nodule metadata
        total_nodule_metadata = np.take(total_nodule_metadata, keep_indices)
        volume_info['volume_array'], volume_info['total_nodule_metadata'] = volume, total_nodule_metadata
        return volume_info

    @classmethod
    def convert_label_value(cls, volume_info):
        volume, total_nodule_metadata = volume_info['volume_array'], volume_info['total_nodule_metadata']
        new_volume = np.zeros_like(volume)
        for idx, nodule_metadata in enumerate(total_nodule_metadata, 1):
            new_volume[volume==nodule_metadata['Nodule_id']] = idx
            nodule_metadata['Nodule_id'] = idx
        volume_info['volume_array'], volume_info['total_nodule_metadata'] = volume, total_nodule_metadata
        return volume_info

    @staticmethod
    def build_nodule_metadata(volume):
        if np.sum(volume) == np.sum(np.zeros_like(volume)):
            return None

        nodule_category = np.unique(volume)
        nodule_category = np.delete(nodule_category, np.where(nodule_category==0))
        total_nodule_metadata = []
        for label in nodule_category:
            binary_mask = volume==label
            nodule_size = np.sum(binary_mask)
            zs, ys, xs = np.where(binary_mask)
            nodule_metadata = {'Nodule_id': label,
                               'Nodule_size': nodule_size,
                               'Nodule_slice': (np.min(zs), np.max(zs))}
            total_nodule_metadata.append(nodule_metadata)
        return total_nodule_metadata
