import os
from detectron2.data.catalog import Metadata
import numpy as np
import matplotlib.pyplot as plt
import cc3d
from utils import utils
import time

# TODO: delte these two globals
CONNECTIVITY = 26
AREA_THRESHOLD = 20



class volumetric_data_eval2():
    def __init__(self, model_name, save_path, dataset_name, match_threshold=0.5, max_nodule_num=1):
        self.model_name = model_name
        self.save_path = save_path
        self.dataset_name = dataset_name
        self.match_threshold = match_threshold
        self.max_nodule_num = max_nodule_num
        self.PixelTP, self.PixelFP, self.PixelFN = [], [] ,[]
        self.VoxelTP, self.VoxelFP, self.VoxelFN = [], [] ,[]
        self.num_nodule = 0
        self.num_case = 0
        self.eval_file = open(os.path.join(self.save_path, 'evaluation.txt'), 'w+')

    def calculate(self, target_study, pred_study):
        self.num_case += 1
        self.num_nodule += len(target_study.nodule_instances)
        assert np.shape(target_study.category_volume) == np.shape(pred_study.category_volume)
        self._3D_evaluation(target_study, pred_study)
        self._2D_evaluation(target_study, pred_study)
    
    def _3D_evaluation(self, target_study, pred_study):
        target_nodules = target_study.nodule_instances
        pred_nodules = pred_study.nodule_instances

        tp, fp, fn = 0, 0, 0
        for target_nodule_id in target_nodules:
            target_nodule = target_nodules[target_nodule_id]
            NoduleIoU, NoduleDSC = [], []
            match_candidates = []
            for pred_nodule_id in pred_nodules:
                pred_nodule = pred_nodules[pred_nodule_id]
                iou = self.BinaryIoU(target_nodule.nodule_volume, pred_nodule.nodule_volume)
                dsc = self.BinaryDSC(target_nodule.nodule_volume, pred_nodule.nodule_volume)

                pred_nodule.add_score('IoU', iou)
                pred_nodule.add_score('DSC', dsc)

                if iou == 0:
                    fp += 1
                else:
                    match_candidates.append(pred_nodule)

            # TODO: input matching function
            merge_nodule = sum([candidate.nodule_volume for candidate in match_candidates])
            merge_nodule = np.where(merge_nodule>0, 1, 0)
            NoduleIoU = self.BinaryIoU(target_nodule.nodule_volume, merge_nodule)
            NoduleDSC = self.BinaryDSC(target_nodule.nodule_volume, merge_nodule)
            target_nodule.add_score('IoU', NoduleIoU)
            target_nodule.add_score('DSC', NoduleDSC)
            if NoduleIoU >= self.match_threshold:
                tp += 1
            else:
                fn += 1

        # target_study.add_score('Volume IoU', NoduleIoU)
        # target_study.add_score('Volume DSC', NoduleIoU)
        # target_study.add_score('Nodule IoU', NoduleIoU)
        # target_study.add_score('Nodule DSC', NoduleDSC)
        target_study.add_score('Nodule TP', tp)
        target_study.add_score('Nodule FP', fp)
        target_study.add_score('Nodule FN', fn)

        self.VoxelTP.append(tp)
        self.VoxelFP.append(fp)
        self.VoxelFN.append(fn)
        print(tp, fp, fn)

    def _2D_evaluation(self, target_study, pred_study):
        binary_target_vol = target_study.get_binary_volume()
        binary_pred_vol = pred_study.get_binary_volume()
        tp = np.sum(np.logical_and(binary_target_vol, binary_pred_vol))
        fp = np.sum(np.logical_and(np.logical_xor(binary_target_vol, binary_pred_vol), binary_pred_vol))
        fn = np.sum(np.logical_and(np.logical_xor(binary_target_vol, binary_pred_vol), binary_target_vol))

        iou = self.BinaryIoU(binary_target_vol, binary_pred_vol)
        dsc = self.BinaryDSC(binary_target_vol, binary_pred_vol)

        pred_study.add_score('Voxel IoU', iou)
        pred_study.add_score('Voxel DSC', dsc)

        self.PixelTP.append(tp)
        self.PixelFP.append(fp)
        self.PixelFN.append(fn)

    # @classmethod
    # def get_submission(cls, target_vol, pred_vol, vol_infos, connectivity=CONNECTIVITY, area_threshold=AREA_THRESHOLD):
    #     target_vol, target_metadata = cls.volume_preprocess(target_vol, connectivity, area_threshold)
    #     pred_vol, pred_metadata = cls.volume_preprocess(pred_vol, connectivity, area_threshold)
    #     pred_category = np.unique(pred_vol)[1:]
    #     total_nodule_infos = []
    #     for label in pred_category:
    #         pred_nodule = np.where(pred_vol==label, 1, 0)
    #         target_nodule = np.logical_or(target_vol>0, pred_nodule)*target_vol
    #         nodule_dsc = cls.BinaryDSC(target_nodule, pred_nodule)
    #         pred_center_irc = utils.get_nodule_center(pred_nodule)
    #         pred_center_xyz = utils.irc2xyz(pred_center_irc, vol_infos['origin'], vol_infos['spacing'], vol_infos['direction'])
    #         nodule_infos= {'Center_xyz': pred_center_xyz, 'Nodule_prob': nodule_dsc}
    #         total_nodule_infos.append(nodule_infos)
    #     return total_nodule_infos

    @staticmethod
    def BinaryIoU(target, pred):
        intersection = np.sum(np.logical_and(target, pred))
        union = np.sum(np.logical_or(target, pred))
        return intersection/union if union != 0 else 0
    
    @staticmethod
    def BinaryDSC(target, pred):
        intersection = np.sum(np.logical_and(target, pred))
        union = np.sum(np.logical_or(target, pred))
        return 2*intersection/(union+intersection) if (union+intersection) != 0 else 0

    def write_and_print(self, message):
        assert isinstance(message, str), 'Message is not string type.'
        self.eval_file.write(message + '\n')
        print(message)

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
            t = time.localtime()
            result = time.strftime("%m/%d/%Y, %H:%M:%S", t)
            self.write_and_print(f'Time: {result}')
            self.write_and_print(f'Model: {self.model_name}')
            self.write_and_print(f'Testing Dataset: {self.dataset_name}')
            self.write_and_print(f'Testing Case: {self.num_case}')
            self.write_and_print(f'Testing Nodule: {self.num_nodule}')
            self.write_and_print(f'Matching Threshold: {self.match_threshold}')
            self.write_and_print('')
            self.write_and_print(f'VoxelTP/Target: {np.sum(self.VoxelTP)}/{np.sum(self.P)}')
            self.write_and_print(f'VoxelTP/Prediction: {np.sum(self.VoxelTP)}/{np.sum(self.VoxelTP)+np.sum(self.VoxelFP)}')
            self.write_and_print(f'Voxel Precision: {self.Volume_Precisiion:.4f}')
            self.write_and_print(f'Voxel Recall: {self.Volume_Recall:.4f}')
            self.write_and_print('')
            self.write_and_print(f'PixelTP/Target: {np.sum(self.PixelTP)}/{np.sum(self.PixelTP)+np.sum(self.PixelFN)}')
            self.write_and_print(f'PixelTP/Prediction: {np.sum(self.PixelTP)}/{np.sum(self.PixelTP)+np.sum(self.PixelFP)}')
            self.write_and_print(f'Pixel Precision: {self.Slice_Precision:.4f}')
            self.write_and_print(f'Pixel Recall: {self.Slice_Recall:.4f}')
            self.write_and_print(f'Pixel F1: {self.Slice_F1:.4f}')
            self.eval_file.close()

        return self.VoxelTP, self.VoxelFP, self.VoxelFN, self.Volume_Precisiion, self.Volume_Recall


class volumetric_data_eval():
    def __init__(self, save_path, connectivity=CONNECTIVITY, area_threshold=AREA_THRESHOLD, match_threshold=0.5, max_nodule_num=1):
        self.connectivity = connectivity
        self.area_threshold = area_threshold
        self.match_threshold = match_threshold
        self.max_nodule_num = max_nodule_num
        self.PixelTP, self.PixelFP, self.PixelFN = [], [] ,[]
        self.VoxelTP, self.VoxelFP, self.VoxelFN = [], [] ,[]
        self.save_path = save_path
        self.eval_file = open(os.path.join(self.save_path, 'evaluation.txt'), 'w+')

    def calculate(self, target_vol, pred_vol, vol_infos=None):
        target_vol, mask_metadata = self.volume_preprocess(target_vol, self.connectivity, self.area_threshold)
        pred_vol, pred_metadata = self.volume_preprocess(pred_vol, self.connectivity, self.area_threshold)
        assert np.shape(target_vol) == np.shape(pred_vol)
        nodule_infos = self._3D_evaluation(target_vol, pred_vol, mask_metadata, pred_metadata, vol_infos)
        self._2D_evaluation(target_vol, pred_vol)
        return nodule_infos
    
    @classmethod
    def volume_preprocess(cls, volume, connectivity=CONNECTIVITY, area_threshold=AREA_THRESHOLD):
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
                                'Relative Size': mask_nodule_metadata['Nodule_size']/target_vol.size,
                                'Depth': target_vol.shape[0]}
                
                gt_nodule_mask = np.logical_and(gt_nodule>0, pred_vol>0)
                pred_nodule_category = np.unique(gt_nodule_mask*pred_vol)[1:]
                if pred_nodule_category.size > 0: # All predictions are wrong if only 0 exist in gt_nodule_mask
                    pred_nodule = sum([pred_vol==label for label in pred_nodule_category])
                    pred_nodule_id = np.min(pred_nodule_category)
                    pred_vol[np.where(pred_nodule)] = pred_nodule_id
                    pred_category = np.delete(pred_category, np.where(pred_category==np.delete(
                            pred_nodule_category, np.where(pred_nodule_category==pred_nodule_id))))
                    pred_nodule_category = np.array([pred_nodule_id])

                    NoduleIoU = self.BinaryIoU(gt_nodule, pred_nodule)
                    NoduleDSC = self.BinaryDSC(gt_nodule, pred_nodule)
                    
                    if NoduleIoU > 0:
                        for slice_idx in range(start_slice_idx, end_slice_idx+1):
                            SliceIOU = self.BinaryIoU(target_vol[slice_idx]==mask_nodule_metadata['Nodule_id'], pred_vol[slice_idx]==pred_nodule_id)
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

                # if NoduleDSC > 0:
                #     pred_center_irc = utils.get_nodule_center(pred_nodule)
                #     pred_center_xyz = utils.irc2xyz(pred_center_irc, vol_infos['origin'], vol_infos['spacing'], vol_infos['direction'])
                # else:
                #     pred_center_xyz = None
                # nodule_infos['Center_xyz'] = pred_center_xyz
                # nodule_infos['Nodule_prob'] = NoduleDSC

                total_nodule_infos.append(nodule_infos)
        else:
            tp = 0

        fp = len(pred_category) if pred_category is not None else 0
        fn = len(mask_category) if mask_category is not None else 0
        self.VoxelTP.append(tp)
        self.VoxelFP.append(fp)
        self.VoxelFN.append(fn)

        print(tp, fp, fn)
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
            nodule_dsc = cls.BinaryDSC(target_nodule, pred_nodule)
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

    @staticmethod
    def BinaryIoU(target, pred):
        intersection = np.sum(np.logical_and(target, pred))
        union = np.sum(np.logical_or(target, pred))
        return intersection/union if union != 0 else 0
    
    @staticmethod
    def BinaryDSC(target, pred):
        intersection = np.sum(np.logical_and(target, pred))
        union = np.sum(np.logical_or(target, pred))
        return 2*intersection/(union+intersection) if (union+intersection) != 0 else 0

    def write_and_print(self, message):
        assert isinstance(message, str), 'Message is not string type.'
        self.eval_file.write(message + '\n')
        print(message)

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
            self.write_and_print(f'Area Threshold: {self.area_threshold}')
            self.write_and_print(f'VoxelTP / Target: {np.sum(self.VoxelTP)} / {np.sum(self.P)}')
            self.write_and_print(f'VoxelFN / Target: {np.sum(self.VoxelFN)} / {np.sum(self.P)}')
            self.write_and_print(f'VoxelTP / Prediction: {np.sum(self.VoxelTP)} / {np.sum(self.VoxelTP)+np.sum(self.VoxelFP)}')
            self.write_and_print(f'VoxelFP / Prediction: {np.sum(self.VoxelFP)} / {np.sum(self.VoxelTP)+np.sum(self.VoxelFP)}')
            self.write_and_print(f'Voxel Precision: {self.Volume_Precisiion:.4f}')
            self.write_and_print(f'Voxel Recall: {self.Volume_Recall:.4f}')
            self.write_and_print('')
            self.write_and_print(f'PixelTP / Target: {np.sum(self.PixelTP)} / {np.sum(self.PixelTP)+np.sum(self.PixelFN)}')
            self.write_and_print(f'PixelFN / Target: {np.sum(self.PixelFN)} / {np.sum(self.PixelTP)+np.sum(self.PixelFN)}')
            self.write_and_print(f'PixelTP / Prediction: {np.sum(self.PixelTP)} / {np.sum(self.PixelTP)+np.sum(self.PixelFP)}')
            self.write_and_print(f'PixelFP / Prediction: {np.sum(self.PixelFP)} / {np.sum(self.PixelTP)+np.sum(self.PixelFP)}')
            self.write_and_print(f'Pixel Precision: {self.Slice_Precision:.4f}')
            self.write_and_print(f'Pixel Recall: {self.Slice_Recall:.4f}')
            self.write_and_print(f'Pixel F1: {self.Slice_F1:.4f}')
            self.eval_file.close()

        return self.VoxelTP, self.VoxelFP, self.VoxelFN, self.Volume_Precisiion, self.Volume_Recall