import numpy as np
import matplotlib.pyplot as plt
import cc3d

class volumetric_data_eval():
    def __init__(self, connectivity=26, area_threshold=20, match_threshold=0.5, max_nodule_num=1):
        self.connectivity = connectivity
        self.area_threshold = area_threshold
        self.match_threshold = match_threshold
        self.max_nodule_num = max_nodule_num
        self.PixelTP, self.PixelFP, self.PixelFN = [], [] ,[]
        self.VoxelTP, self.VoxelFP, self.VoxelFN = [], [] ,[]

    @staticmethod
    def IoU(target, pred):
        # Only input binary array
        intersection = np.sum(np.logical_and(target, pred))
        union = np.sum(np.logical_or(target, pred))
        return intersection/union if union != 0 else 0
    
    @classmethod
    def calculate_nodule_size(cls, volume, area_threshold):
        """Calculate every nodule size in single volume and return nodule sizes bigger than area_threshold"""
        # TODO: separate duty, caluculate size, change volume value
        # _, counts = np.unique(volume, return_counts=True)
        # total_area_size = np.fromiter(counts.values(), dtype=int)[1:]
        # total_area_size = np.delete(total_area_size, np.where(total_area_size<area_threshold)[0])
        # print(np.max(volume))
        category = list(range(1, np.max(volume)+1))
        total_area_size = np.array([], dtype=np.int32)
        for label in range(1, np.max(volume)+1):
            area_size = np.sum(volume==label)
            if area_size < area_threshold:
                volume = np.where(volume==label, 0, volume)
                category.remove(label)
            else:
                total_area_size = np.append(total_area_size, area_size)
        # print(np.max(volume))
        return total_area_size, volume

    # @classmethod
    # def convert_volume_value(cls, volume, ):


    @classmethod
    def remove_small_area(cls, volume, area_threshold, max_nodule_num=None):
        total_area_size, volume = cls.calculate_nodule_size(volume, area_threshold)

        # Descending sort the category based on nodule size
        # category = list(range(1, np.max(volume)+1))
        category = np.unique(volume)
        category = np.delete(category, np.where(category==0))
        category = np.take(category, np.argsort(total_area_size)[::-1])

        # Convert Label and keep big enough nodules (keeping nodule number = max_nodule_num)
        # print(np.max(volume))
        for idx, label in enumerate(category, 1):
            volume = np.where(volume==label, idx, volume)
            if max_nodule_num is not None:
                if idx == max_nodule_num:
                    break
        if max_nodule_num is not None:
            volume = np.where(volume>max_nodule_num, 0, volume)
        # print(np.max(volume))
        return volume
    
    @classmethod
    def mask_and_pred_volume_preprocess(cls, mask_vol, pred_vol, connectivity, area_threshold):
        mask_vol = cc3d.connected_components(mask_vol, connectivity=connectivity)
        pred_vol = cc3d.connected_components(pred_vol, connectivity=connectivity)
        mask_vol = cls.remove_small_area(mask_vol, area_threshold)
        pred_vol = cls.remove_small_area(pred_vol, area_threshold)
        assert np.shape(mask_vol) == np.shape(pred_vol)
        return mask_vol, pred_vol

    def calculate(self, mask_vol, pred_vol):
        # print(np.max(mask_vol))
        mask_vol, pred_vol = self.mask_and_pred_volume_preprocess(mask_vol, pred_vol, self.connectivity, self.area_threshold)
        # print(np.max(mask_vol))
        self._3D_evaluation(mask_vol, pred_vol)
        self._2D_evaluation(mask_vol, pred_vol)

    def _2D_evaluation(self, mask_vol, pred_vol):
        binary_mask_vol = np.where(mask_vol>0, 1, 0)
        binary_pred_vol = np.where(pred_vol>0, 1, 0)
        iou = self.IoU(binary_mask_vol, binary_pred_vol)
        tp = np.sum(np.logical_and(binary_mask_vol, binary_pred_vol))
        fp = np.sum(np.logical_and(np.logical_xor(binary_mask_vol, binary_pred_vol), binary_pred_vol))
        fn = np.sum(np.logical_and(np.logical_xor(binary_mask_vol, binary_pred_vol), binary_mask_vol))
        self.PixelTP.append(tp)
        self.PixelFP.append(fp)
        self.PixelFN.append(fn)

    def _3D_evaluation(self, mask_vol, pred_vol):
        tp, fp, fn = 0, 0, 0
        # print(np.max(mask_vol))
        mask_category = list(range(1, np.max(mask_vol)+1))
        pred_category = list(range(1, np.max(pred_vol)+1))
        for mask_label in mask_category:
            for pred_label in pred_category:
                IntersectionOverUinion = self.IoU(mask_vol==mask_label, pred_vol==pred_label)
                # zs, ys, xs = np.where(mask_vol==mask_label)
                # unique_zs = np.unique(zs)
                # print('Slice', unique_zs)
                # print(f'Slice from {np.min(zs)} to {np.max(zs)}')
                if IntersectionOverUinion >= self.match_threshold:
                    tp += 1
                    mask_category.remove(mask_label)
                    pred_category.remove(pred_label)
                    break
        fp = len(pred_category)
        fn = len(mask_category)
        self.VoxelTP.append(tp)
        self.VoxelFP.append(fp)
        self.VoxelFN.append(fn)

    def evaluation(self, show_evaluation):
        # TODO: remove repeat part
        self.VoxelTP = np.array(self.VoxelTP)
        self.VoxelFP = np.array(self.VoxelFP)
        self.VoxelFN = np.array(self.VoxelFN)
        self.P = self.VoxelTP + self.VoxelFN
        precision = self.VoxelTP / (self.VoxelTP + self.VoxelFP)
        recall = self.VoxelTP / (self.VoxelTP + self.VoxelFN)
        self.Precision = np.where(np.isnan(precision), 0, precision)
        self.Recall = np.where(np.isnan(recall), 0, recall)
        self.System_Precisiion = np.sum(self.VoxelTP) / (np.sum(self.VoxelTP) + np.sum(self.VoxelFP))
        self.System_Recall = np.sum(self.VoxelTP) / np.sum(self.P)
        self.Pixel_Precision = np.sum(self.PixelTP) / (np.sum(self.PixelTP) + np.sum(self.PixelFP))
        self.Pixel_Recall = np.sum(self.PixelTP) / (np.sum(self.PixelTP) + np.sum(self.PixelFN))
        self.Pixel_F1 = 2*np.sum(self.PixelTP) / (2*np.sum(self.PixelTP) + np.sum(self.PixelFP) + np.sum(self.PixelFN))

        if show_evaluation:
            print(f'Area Threshold: {self.area_threshold}')
            print(f'VoxelTP / Target: {np.sum(self.VoxelTP)} / {np.sum(self.P)}')
            print(f'VoxelFN / Target: {np.sum(self.VoxelFN)} / {np.sum(self.P)}')
            print(f'VoxelTP / Prediction: {np.sum(self.VoxelTP)} / {np.sum(self.VoxelTP)+np.sum(self.VoxelFP)}')
            print(f'VoxelFP / Prediction: {np.sum(self.VoxelFP)} / {np.sum(self.VoxelTP)+np.sum(self.VoxelFP)}')
            print(f'System Precision: {self.System_Precisiion:.4f}')
            print(f'System Recall: {self.System_Recall:.4f}')
            print('')
            print(f'PixelTP / Target: {np.sum(self.PixelTP)} / {np.sum(self.PixelTP)+np.sum(self.PixelFN)}')
            print(f'PixelFN / Target: {np.sum(self.PixelFN)} / {np.sum(self.PixelTP)+np.sum(self.PixelFN)}')
            print(f'PixelTP / Prediction: {np.sum(self.PixelTP)} / {np.sum(self.PixelTP)+np.sum(self.PixelFP)}')
            print(f'PixelFP / Prediction: {np.sum(self.PixelFP)} / {np.sum(self.PixelTP)+np.sum(self.PixelFP)}')
            print(f'Pixel Precision: {self.Pixel_Precision:.4f}')
            print(f'Pixel Recall: {self.Pixel_Recall:.4f}')
            print(f'Pixel F1: {self.Pixel_F1:.4f}')
            print('')
            print(f'mean Precision: {np.mean(self.Precision):.4f}')
            print(f'mean Recall: {np.mean(self.Recall):.4f}')
        return self.VoxelTP, self.VoxelFP, self.VoxelFN, self.System_Precisiion, self.System_Recall