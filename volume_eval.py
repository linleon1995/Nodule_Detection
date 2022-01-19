from detectron2.data.catalog import Metadata
import numpy as np
import matplotlib.pyplot as plt
import cc3d
import utils
CONNECTIVITY = 26
AREA_THRESHOLD = 20

class volumetric_data_eval():
    def __init__(self, connectivity=CONNECTIVITY, area_threshold=AREA_THRESHOLD, match_threshold=0.5, max_nodule_num=1):
        self.connectivity = connectivity
        self.area_threshold = area_threshold
        self.match_threshold = match_threshold
        self.max_nodule_num = max_nodule_num
        self.PixelTP, self.PixelFP, self.PixelFN = [], [] ,[]
        self.VoxelTP, self.VoxelFP, self.VoxelFN = [], [] ,[]

    def calculate(self, target_vol, pred_vol, vol_infos):
        target_vol, mask_metadata = self.volume_preprocess(target_vol, self.connectivity, self.area_threshold)
        pred_vol, pred_metadata = self.volume_preprocess(pred_vol, self.connectivity, self.area_threshold)
        assert np.shape(target_vol) == np.shape(pred_vol)
        nodule_infos = self._3D_evaluation(target_vol, pred_vol, mask_metadata, pred_metadata, vol_infos)
        self._2D_evaluation(target_vol, pred_vol)
        return nodule_infos
    
    @classmethod
    def volume_preprocess(cls, volume, connectivity, area_threshold):
        volume = cc3d.connected_components(volume, connectivity=connectivity)
        total_nodule_metadata = cls.build_nodule_metadata(volume)
        if total_nodule_metadata is not None:
            volume, total_nodule_metadata = cls.remove_small_area(volume, total_nodule_metadata, area_threshold)
            volume, total_nodule_metadata = cls.convert_label_value(volume, total_nodule_metadata)
        return volume, total_nodule_metadata

    def _3D_evaluation(self, target_vol, pred_vol, mask_metadata, pred_metadata, vol_infos):
        # TODO: Distinduish nodule_metadata and nodule_infos
        tp, fp, fn = 0, 0, 0
        mask_category = list(range(1, np.max(target_vol)+1)) if np.max(target_vol) > 0 else None
        pred_category = list(range(1, np.max(pred_vol)+1)) if np.max(pred_vol) > 0 else None
        total_nodule_infos = []

        if mask_category is not None and pred_category is not None:
            for mask_nodule_metadata in mask_metadata:
                gt_nodule = np.int32(target_vol==mask_nodule_metadata['Nodule_id'])
                start_slice_idx, end_slice_idx = mask_nodule_metadata['Nodule_slice']
                num_slice = end_slice_idx - start_slice_idx + 1
                BestSliceIoU, BestSliceIndex = 0, 'Null'
                NoduleIoU, NoduleDSC = 0, 0
                nodule_infos = {'Nodule ID': np.int32(mask_nodule_metadata['Nodule_id']),
                                'Slice Number': np.int32(num_slice), 
                                'Size': np.int32(mask_nodule_metadata['Nodule_size']), 
                                'Relative Size': mask_nodule_metadata['Nodule_size']/target_vol.size}
                
                gt_nodule_mask = np.logical_and(gt_nodule>0, pred_vol>0)
                pred_nodule_category = np.unique(gt_nodule_mask*pred_vol)[1:]
                if pred_nodule_category.size > 0: # All predictions are wrong if only 0 exist in gt_nodule_mask
                    pred_nodule = sum([pred_vol==label for label in pred_nodule_category])
                    pred_nodule_id = np.min(pred_nodule_category)
                    pred_vol[np.where(pred_nodule)] = pred_nodule_id
                    pred_category = np.delete(pred_category, np.where(pred_category==np.delete(
                            pred_nodule_category, np.where(pred_nodule_category==pred_nodule_id))))
                    pred_nodule_category = np.array([pred_nodule_id])

                    NoduleIoU = self.IoU(gt_nodule, pred_nodule)
                    NoduleDSC = self.DSC(gt_nodule, pred_nodule)
                    
                    if NoduleIoU > 0:
                        for slice_idx in range(start_slice_idx, end_slice_idx+1):
                            SliceIOU = self.IoU(target_vol[slice_idx]==mask_nodule_metadata['Nodule_id'], pred_vol[slice_idx]==pred_nodule_id)
                            if SliceIOU > BestSliceIoU:
                                BestSliceIoU = SliceIOU
                                BestSliceIndex = slice_idx
                        if NoduleIoU >= self.match_threshold:
                            tp += 1
                            mask_category.remove(mask_nodule_metadata['Nodule_id'])
                            pred_category = np.delete(pred_category, np.where(pred_category==pred_nodule_id))
                            # break

                nodule_infos['Nodule IoU'] = NoduleIoU
                nodule_infos['Nodule DSC'] = NoduleDSC
                nodule_infos['Best Slice IoU'] = BestSliceIoU
                nodule_infos['Best Slice Index'] = BestSliceIndex

                if NoduleDSC > 0:
                    pred_center_irc = utils.get_nodule_center(pred_nodule)
                    pred_center_xyz = utils.irc2xyz(pred_center_irc, vol_infos['origin'], vol_infos['spacing'], vol_infos['direction'])
                else:
                    pred_center_xyz = None
                nodule_infos['Center_xyz'] = pred_center_xyz
                nodule_infos['Nodule_prob'] = NoduleDSC

                total_nodule_infos.append(nodule_infos)
        else:
            tp = 0

        fp = len(pred_category) if pred_category is not None else 0
        fn = len(mask_category) if mask_category is not None else 0
        self.VoxelTP.append(tp)
        self.VoxelFP.append(fp)
        self.VoxelFN.append(fn)
        return total_nodule_infos
    
    def _2D_evaluation(self, target_vol, pred_vol):
        binary_target_vol = np.where(target_vol>0, 1, 0)
        binary_pred_vol = np.where(pred_vol>0, 1, 0)
        tp = np.sum(np.logical_and(binary_target_vol, binary_pred_vol))
        fp = np.sum(np.logical_and(np.logical_xor(binary_target_vol, binary_pred_vol), binary_pred_vol))
        fn = np.sum(np.logical_and(np.logical_xor(binary_target_vol, binary_pred_vol), binary_target_vol))
        self.PixelTP.append(tp)
        self.PixelFP.append(fp)
        self.PixelFN.append(fn)

    @classmethod
    def get_submission(cls, target_vol, pred_vol, vol_infos, connectivity=CONNECTIVITY, area_threshold=AREA_THRESHOLD):
        target_vol, target_metadata = cls.volume_preprocess(target_vol, connectivity, area_threshold)
        pred_vol, pred_metadata = cls.volume_preprocess(pred_vol, connectivity, area_threshold)
        pred_category = np.unique(pred_vol)[1:]
        total_nodule_infos = []
        for label in pred_category:
            pred_nodule = np.where(pred_vol==label, 1, 0)
            target_nodule = np.logical_or(target_vol>0, pred_nodule)*target_vol
            nodule_dsc = cls.DSC(target_nodule, pred_nodule)
            pred_center_irc = utils.get_nodule_center(pred_nodule)
            pred_center_xyz = utils.irc2xyz(pred_center_irc, vol_infos['origin'], vol_infos['spacing'], vol_infos['direction'])
            nodule_infos= {'Center_xyz': pred_center_xyz, 'Nodule_prob': nodule_dsc}
            total_nodule_infos.append(nodule_infos)
        return total_nodule_infos

    @classmethod
    def remove_small_area(cls, volume, total_nodule_metadata, area_threshold):
        keep_indices = list(range(len(total_nodule_metadata)))
        for idx, nodule_metadata in enumerate(total_nodule_metadata):
            if nodule_metadata['Nodule_size'] < area_threshold:
                keep_indices.remove(idx)
                volume[volume==nodule_metadata['Nodule_id']] = 0

        # Remove smaller nodule metadata
        total_nodule_metadata = np.take(total_nodule_metadata, keep_indices)
        return volume, total_nodule_metadata

    @classmethod
    def convert_label_value(cls, volume, total_nodule_metadata):
        new_volume = np.zeros_like(volume)
        for idx, nodule_metadata in enumerate(total_nodule_metadata, 1):
            new_volume[volume==nodule_metadata['Nodule_id']] = idx
            nodule_metadata['Nodule_id'] = idx
        return new_volume, total_nodule_metadata

    @staticmethod
    def build_nodule_metadata(volume):
        # TODO: make sure input is bool or int array
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
            # nodule_metadata['Nodule_id'] = label
            # nodule_metadata['Nodule_size'] = nodule_size
            # nodule_metadata['Nodule_slice'] = (np.min(zs), np.max(zs))
            total_nodule_metadata.append(nodule_metadata)
        return total_nodule_metadata

    @staticmethod
    def IoU(target, pred):
        # TODO: Add constraint for dtype or build multi-class version
        # Only input binary array
        intersection = np.sum(np.logical_and(target, pred))
        union = np.sum(np.logical_or(target, pred))
        return intersection/union if union != 0 else 0
    
    @staticmethod
    def DSC(target, pred):
        intersection = np.sum(np.logical_and(target, pred))
        union = np.sum(np.logical_or(target, pred))
        return 2*intersection/(union+intersection) if (union+intersection) != 0 else 0

    def evaluation(self, show_evaluation):
        self.VoxelTP = np.array(self.VoxelTP)
        self.VoxelFP = np.array(self.VoxelFP)
        self.VoxelFN = np.array(self.VoxelFN)
        self.P = self.VoxelTP + self.VoxelFN

        precision = self.VoxelTP / (self.VoxelTP + self.VoxelFP)
        recall = self.VoxelTP / (self.VoxelTP + self.VoxelFN)
        self.Precision = np.where(np.isnan(precision), 0, precision)
        self.Recall = np.where(np.isnan(recall), 0, recall)
        self.Volume_Precisiion = np.sum(self.VoxelTP) / (np.sum(self.VoxelTP) + np.sum(self.VoxelFP))
        self.Volume_Recall = np.sum(self.VoxelTP) / np.sum(self.P)
        self.Slice_Precision = np.sum(self.PixelTP) / (np.sum(self.PixelTP) + np.sum(self.PixelFP))
        self.Slice_Recall = np.sum(self.PixelTP) / (np.sum(self.PixelTP) + np.sum(self.PixelFN))
        self.Slice_F1 = 2*np.sum(self.PixelTP) / (2*np.sum(self.PixelTP) + np.sum(self.PixelFP) + np.sum(self.PixelFN))

        if show_evaluation:
            print(f'Area Threshold: {self.area_threshold}')
            print(f'VoxelTP / Target: {np.sum(self.VoxelTP)} / {np.sum(self.P)}')
            print(f'VoxelFN / Target: {np.sum(self.VoxelFN)} / {np.sum(self.P)}')
            print(f'VoxelTP / Prediction: {np.sum(self.VoxelTP)} / {np.sum(self.VoxelTP)+np.sum(self.VoxelFP)}')
            print(f'VoxelFP / Prediction: {np.sum(self.VoxelFP)} / {np.sum(self.VoxelTP)+np.sum(self.VoxelFP)}')
            print(f'System Precision: {self.Volume_Precisiion:.4f}')
            print(f'System Recall: {self.Volume_Recall:.4f}')
            print('')
            print(f'PixelTP / Target: {np.sum(self.PixelTP)} / {np.sum(self.PixelTP)+np.sum(self.PixelFN)}')
            print(f'PixelFN / Target: {np.sum(self.PixelFN)} / {np.sum(self.PixelTP)+np.sum(self.PixelFN)}')
            print(f'PixelTP / Prediction: {np.sum(self.PixelTP)} / {np.sum(self.PixelTP)+np.sum(self.PixelFP)}')
            print(f'PixelFP / Prediction: {np.sum(self.PixelFP)} / {np.sum(self.PixelTP)+np.sum(self.PixelFP)}')
            print(f'Pixel Precision: {self.Slice_Precision:.4f}')
            print(f'Pixel Recall: {self.Slice_Recall:.4f}')
            print(f'Pixel F1: {self.Slice_F1:.4f}')
            print('')
            print(f'mean Precision: {np.mean(self.Precision):.4f}')
            print(f'mean Recall: {np.mean(self.Recall):.4f}')
        return self.VoxelTP, self.VoxelFP, self.VoxelFN, self.Volume_Precisiion, self.Volume_Recall